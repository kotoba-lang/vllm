(ns vllm.tokenizer-test
  (:require [clojure.test :refer [deftest is]] [vllm.tokenizer :as tokenizer]))

(def metadata
  {"tokenizer.ggml.model" "llama"
   "tokenizer.ggml.tokens" ["<unk>" "<s>" "</s>" "▁hello" "▁world" "!" "▁" "world"]
   "tokenizer.ggml.scores" [-10 -10 -10 2 2 1 -2 0]
   "tokenizer.ggml.token_type" [2 3 3 1 1 1 1 1]
   "tokenizer.ggml.bos_token_id" 1
   "tokenizer.ggml.eos_token_id" 2
   "tokenizer.ggml.unknown_token_id" 0})

(deftest sentencepiece-viterbi-roundtrip
  (let [t (tokenizer/from-metadata metadata)
        ids (tokenizer/encode t "hello world!" {:add-eos? true})]
    (is (= [1 3 4 5 2] ids))
    (is (= "hello world!" (tokenizer/decode t ids)))))
