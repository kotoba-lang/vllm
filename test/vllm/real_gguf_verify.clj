(ns vllm.real-gguf-verify
  (:require [vllm.generate :as generate]
            [vllm.gguf :as gguf]
            [vllm.llama :as llama]
            [vllm.tokenizer :as tokenizer])
  (:gen-class)
  (:import [java.nio.file Files Paths]
           [java.security MessageDigest]))

(defn- sha256 [path]
  (let [digest (MessageDigest/getInstance "SHA-256")]
    (with-open [input (Files/newInputStream (Paths/get path (make-array String 0))
                                            (make-array java.nio.file.OpenOption 0))]
      (let [buffer (byte-array 1048576)]
        (loop []
          (let [n (.read input buffer)]
            (when (pos? n) (.update digest buffer 0 n) (recur))))))
    (apply str (map #(format "%02x" (bit-and 0xff %)) (.digest digest)))))

(defn -main [& [path]]
  (when-not path
    (throw (ex-info "usage: clojure -M:real-gguf-verify /path/model.gguf" {})))
  (let [started (System/nanoTime)]
    (with-open [file (gguf/open-file path)]
      (let [model (llama/load-gguf file)
            tokenizer (tokenizer/from-metadata (:metadata file))
            prompt "Once upon a time"
            prompt-ids (tokenizer/encode tokenizer prompt)
            result (generate/generate-tokens
                    model (llama/new-state model) prompt-ids
                    {:max-tokens 8 :temperature 0
                     :eos-token-ids #{(:eos-id tokenizer)}})
            text (tokenizer/decode tokenizer (:tokens result))
            elapsed (/ (- (System/nanoTime) started) 1.0e9)]
        (when-not (every? #(Float/isFinite %) (:logits
                                               (llama/step! model
                                                            (llama/new-state model)
                                                            (first prompt-ids))))
          (throw (ex-info "real GGUF produced non-finite logits" {})))
        (println (pr-str {:sha256 (sha256 path)
                          :architecture (get (:metadata file) "general.architecture")
                          :prompt-tokens (count prompt-ids)
                          :generated-tokens (count (:tokens result))
                          :text text :seconds elapsed}))))))
