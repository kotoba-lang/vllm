(ns vllm.gguf-test
  (:require [clojure.test :refer [deftest is]] [vllm.gguf :as gguf])
  (:import [java.nio ByteBuffer ByteOrder]
           [java.nio.file Files OpenOption]
           [java.nio.file.attribute FileAttribute]))

(defn- put-string! [^ByteBuffer b s]
  (let [xs (.getBytes s "UTF-8")]
    (.putLong b (long (alength xs))) (.put b xs)))

(defn- fixture []
  (let [b (doto (ByteBuffer/allocate 512) (.order ByteOrder/LITTLE_ENDIAN))]
    (.put b (.getBytes "GGUF" "US-ASCII")) (.putInt b 3) (.putLong b 2) (.putLong b 3)
    (put-string! b "general.architecture") (.putInt b 8) (put-string! b "llama")
    (put-string! b "general.alignment") (.putInt b 4) (.putInt b 32)
    (put-string! b "tokenizer.ggml.tokens") (.putInt b 9) (.putInt b 8)
    (.putLong b 3) (doseq [s ["a" "b" "c"]] (put-string! b s))
    (put-string! b "token_embd.weight") (.putInt b 2) (.putLong b 2) (.putLong b 3)
    (.putInt b 0) (.putLong b 0)
    (put-string! b "output_norm.weight") (.putInt b 1) (.putLong b 2)
    (.putInt b 1) (.putLong b 32)
    (while (not (zero? (mod (.position b) 32))) (.put b (byte 0)))
    (let [data-start (.position b)]
      (doseq [x [1 2 3 4 5 6]] (.putFloat b (float x)))
      (while (< (.position b) (+ data-start 32)) (.put b (byte 0)))
      (.putShort b (unchecked-short 0x3c00)) (.putShort b (unchecked-short 0x4000)))
    (java.util.Arrays/copyOf (.array b) (.position b))))

(deftest parses-v3-metadata-catalog-and-lazy-windows
  (let [path (Files/createTempFile "vllm-" ".gguf" (make-array FileAttribute 0))]
    (try
      (Files/write path (fixture) (make-array OpenOption 0))
      (with-open [model (gguf/open-file path)]
        (is (= "llama" (get (:metadata model) "general.architecture")))
        (is (= ["a" "b" "c"] (get (:metadata model) "tokenizer.ggml.tokens")))
        (is (= ["output_norm.weight" "token_embd.weight"] (gguf/tensor-names model)))
        (is (= [3 2] (:shape (gguf/tensor-info model "token_embd.weight"))))
        (is (= 24 (.remaining (gguf/read-tensor-bytes model "token_embd.weight"))))
        (is (= 4 (.remaining (gguf/read-tensor-bytes model "output_norm.weight"))))
        (is (= [1.0 2.0 3.0 4.0 5.0 6.0]
               (vec (:data (gguf/read-tensor-f32 model "token_embd.weight")))))
        (is (= [1.0 2.0]
               (vec (:data (gguf/read-tensor-f32 model "output_norm.weight"))))))
      (finally (Files/deleteIfExists path)))))

(defn- little-buffer [size]
  (doto (ByteBuffer/allocate size) (.order ByteOrder/LITTLE_ENDIAN)))

(deftest decodes-common-quantized-blocks
  (let [q4-0 (little-buffer 18)
        _ (.putShort q4-0 (unchecked-short 0x3c00))
        _ (dotimes [i 16] (.put q4-0 (unchecked-byte
                                      (bit-or i (bit-shift-left (- 15 i) 4)))))
        q4-1 (little-buffer 20)
        _ (.putShort q4-1 (unchecked-short 0x4000))
        _ (.putShort q4-1 (unchecked-short 0xbc00))
        _ (dotimes [i 16] (.put q4-1 (unchecked-byte
                                      (bit-or i (bit-shift-left i 4)))))
        q8-0 (little-buffer 34)
        _ (.putShort q8-0 (unchecked-short 0x3800))
        _ (doseq [i (range -16 16)] (.put q8-0 (byte i)))]
    (.flip q4-0) (.flip q4-1) (.flip q8-0)
    (is (= (mapv float (concat (range -8 8) (range 7 -9 -1)))
           (vec (gguf/decode-f32 :q4-0 q4-0 32))))
    (is (= (mapv float (concat (map #(- (* 2 %) 1) (range 16))
                               (map #(- (* 2 %) 1) (range 16))))
           (vec (gguf/decode-f32 :q4-1 q4-1 32))))
    (is (= (mapv #(float (* 0.5 %)) (range -16 16))
           (vec (gguf/decode-f32 :q8-0 q8-0 32))))))
