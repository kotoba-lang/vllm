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
          (throw (ex-info "Metal worker leaked its handle" {}))))
      (finally (.close ^Closeable worker)))))
