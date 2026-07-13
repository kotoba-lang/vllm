(ns vllm.metal-worker-verify
  "Live JVM -> Deno -> WebGPU/Metal persistent Q8_0 verification."
  (:require [vllm.accelerator :as accelerator]
            [vllm.metal-worker :as metal])
  (:import [java.io Closeable]
           [java.nio ByteBuffer ByteOrder]))

(defn- raw-q8-matrix []
  (let [scales [0x3800 0x3c00 0x3e00 0x3400]
        bytes (doto (ByteBuffer/allocate (* 4 34)) (.order ByteOrder/LITTLE_ENDIAN))]
    (doseq [scale scales]
      (.putShort bytes (short scale))
      (doseq [q (range -16 16)] (.put bytes (byte q))))
    (.flip bytes)))

(defn- raw-q4-matrix []
  (doto (ByteBuffer/allocate 36)
    (.order ByteOrder/LITTLE_ENDIAN)
    (.putShort (short 0x3800))
    (.put (byte-array 16 (unchecked-byte 0x98)))
    (.putShort (short 0x3c00))
    (.put (byte-array 16 (unchecked-byte 0x98)))
    (.flip)))

(defn- raw-q4k-matrix []
  (doto (ByteBuffer/allocate 144)
    (.order ByteOrder/LITTLE_ENDIAN)
    (.putShort (short 0x3c00))
    (.putShort (short 0x3c00))
    (.put (byte-array (concat (repeat 8 1) (repeat 4 0x11))))
    (.put (byte-array 128))
    (.flip)))

(defn- raw-q5-matrix []
  (doto (ByteBuffer/allocate 44)
    (.order ByteOrder/LITTLE_ENDIAN)
    (.putShort (short 0x3c00)) (.putInt (unchecked-int 0xffffffff))
    (.put (byte-array 16 (unchecked-byte 0x10)))
    (.putShort (short 0x3800)) (.putInt (unchecked-int 0xffffffff))
    (.put (byte-array 16 (unchecked-byte 0x10)))
    (.flip)))

(defn- raw-q5-1-matrix []
  (doto (ByteBuffer/allocate 48)
    (.order ByteOrder/LITTLE_ENDIAN)
    (.putShort (short 0x3c00)) (.putShort (short 0x3c00)) (.putInt 0)
    (.put (byte-array 16 (unchecked-byte 0x10)))
    (.putShort (short 0x3c00)) (.putShort (short 0x3c00)) (.putInt 0)
    (.put (byte-array 16 (unchecked-byte 0x10)))
    (.flip)))

(defn- raw-q5k-matrix []
  (doto (ByteBuffer/allocate 176)
    (.order ByteOrder/LITTLE_ENDIAN)
    (.putShort (short 0x3c00)) (.putShort (short 0x3c00))
    (.put (byte-array (concat (repeat 8 1) (repeat 4 0x11))))
    (.put (byte-array 32)) (.put (byte-array 128))
    (.flip)))

(defn- raw-q6k-matrix []
  (doto (ByteBuffer/allocate 210)
    (.order ByteOrder/LITTLE_ENDIAN)
    (.put (byte-array 128))
    (.put (byte-array 64 (unchecked-byte 0xaa)))
    (.put (byte-array 16 (byte 1)))
    (.putShort (short 0x3c00))
    (.flip)))

(defn- close? [expected actual]
  (every? true? (map #(< (Math/abs (- %1 %2)) 1.0e-4) expected actual)))

(defn -main [& [worker-path]]
  (let [path (or worker-path "../num/target/deno-q8-worker.cjs")
        worker (metal/start ["deno" "run" "--allow-all" path])]
    (try
      (let [handle (accelerator/upload-q8! worker "verify" 2 64 (raw-q8-matrix))
            x (float-array (map #(- (* 0.02 %) 0.4) (range 64)))
            first-result (vec (accelerator/gemv! worker handle x))
            doubled (float-array (map #(* 2 %) x))
            second-result (vec (accelerator/gemv! worker handle doubled))
            batched (mapv vec (accelerator/gemv-many! worker [[handle x] [handle doubled]]))]
        (println "Metal Q8_0 first:" first-result "second:" second-result)
        (when-not (and (close? [73.76 95.44] first-result)
                       (close? [147.52 190.88] second-result)
                       (= [first-result second-result] batched)
                       (= 1 (:handles (metal/stats worker))))
          (throw (ex-info "Metal worker verification failed" {})))
        (accelerator/release! worker handle)
        (when-not (zero? (:handles (metal/stats worker)))
          (throw (ex-info "Metal worker leaked its handle" {})))
        (let [q4 (accelerator/upload-quantized! worker :q4-0 "q4" 2 32
                                                 (raw-q4-matrix))
              result (vec (accelerator/gemv! worker q4 (float-array (repeat 32 1.0))))]
          (println "Metal Q4_0:" result)
          (when-not (close? [8.0 16.0] result)
            (throw (ex-info "Metal Q4_0 verification failed" {:actual result})))
          (accelerator/release! worker q4))
        (let [q4k (accelerator/upload-quantized! worker :q4-k "q4k" 1 256
                                                  (raw-q4k-matrix))
              result (vec (accelerator/gemv! worker q4k (float-array (repeat 256 1.0))))]
          (println "Metal Q4_K:" result)
          (when-not (close? [-256.0] result)
            (throw (ex-info "Metal Q4_K verification failed" {:actual result})))
          (accelerator/release! worker q4k))
        (doseq [[type raw expected] [[:q5-0 (raw-q5-matrix) 24.0]
                                     [:q5-1 (raw-q5-1-matrix) 96.0]]]
          (let [handle (accelerator/upload-quantized! worker type (name type) 1 64 raw)
                result (vec (accelerator/gemv! worker handle
                                                (float-array (repeat 64 1.0))))]
            (println "Metal" type result)
            (when-not (close? [expected] result)
              (throw (ex-info "Metal Q5 verification failed"
                              {:type type :actual result})))
            (accelerator/release! worker handle)))
        (doseq [[type raw expected] [[:q5-k (raw-q5k-matrix) -256.0]
                                     [:q6-k (raw-q6k-matrix) 0.0]]]
          (let [handle (accelerator/upload-quantized! worker type (name type) 1 256 raw)
                result (vec (accelerator/gemv! worker handle
                                                (float-array (repeat 256 1.0))))]
            (println "Metal" type result)
            (when-not (close? [expected] result)
              (throw (ex-info "Metal K-quant verification failed"
                              {:type type :actual result})))
            (accelerator/release! worker handle))))
      (finally (.close ^Closeable worker)))))
