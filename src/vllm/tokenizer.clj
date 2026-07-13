(ns vllm.tokenizer
  "GGUF-embedded Llama/SentencePiece tokenizer with Viterbi segmentation."
  (:require [clojure.string :as str])
  (:import [java.io ByteArrayOutputStream]
           [java.nio.charset StandardCharsets]))

(defn from-metadata [metadata]
  (let [tokens (get metadata "tokenizer.ggml.tokens")
        scores (or (get metadata "tokenizer.ggml.scores") (repeat (count tokens) 0.0))
        types (or (get metadata "tokenizer.ggml.token_type") (repeat (count tokens) 1))]
    (when-not (and (seq tokens) (= (count tokens) (count scores) (count types)))
      (throw (ex-info "invalid GGUF tokenizer metadata" {})))
    {:model (get metadata "tokenizer.ggml.model")
     :tokens (vec tokens) :scores (mapv double scores) :types (mapv long types)
     :token->id (zipmap tokens (range))
     :bos-id (get metadata "tokenizer.ggml.bos_token_id")
     :eos-id (get metadata "tokenizer.ggml.eos_token_id")
     :unknown-id (get metadata "tokenizer.ggml.unknown_token_id")}))

(defn- normalized [text]
  (str "▁" (str/replace (str text) " " "▁")))

(defn- byte-fallback [tokenizer text]
  (mapv (fn [b]
          (or (get (:token->id tokenizer) (format "<0x%02X>" (bit-and 0xff b)))
              (:unknown-id tokenizer)
              (throw (ex-info "tokenizer has no byte fallback or unknown token"
                              {:byte (bit-and 0xff b)}))))
        (.getBytes text StandardCharsets/UTF_8)))

(defn encode
  "Encode text using maximum-score SentencePiece segmentation. Options:
  `:add-bos?` (default true) and `:add-eos?` (default false)."
  ([tokenizer text] (encode tokenizer text {}))
  ([tokenizer text {:keys [add-bos? add-eos?] :or {add-bos? true add-eos? false}}]
   (let [source (normalized text) n (.length source)
         best (object-array (inc n))]
     (aset best 0 {:score 0.0 :ids []})
     (dotimes [start n]
       (when-let [{:keys [score ids]} (aget best start)]
         (doseq [[id token] (map-indexed vector (:tokens tokenizer))
                 :when (and (not (contains? #{3 5} (nth (:types tokenizer) id)))
                            (.startsWith source token start))]
           (let [end (+ start (.length token)) candidate (+ score (nth (:scores tokenizer) id))
                 previous (aget best end)]
             (when (or (nil? previous) (> candidate (:score previous)))
               (aset best end {:score candidate :ids (conj ids id)}))))))
     (let [ids (if-let [result (aget best n)]
                 (:ids result)
                 (byte-fallback tokenizer source))]
       (cond-> []
         (and add-bos? (some? (:bos-id tokenizer))) (conj (:bos-id tokenizer))
         true (into ids)
         (and add-eos? (some? (:eos-id tokenizer))) (conj (:eos-id tokenizer)))))))

(defn decode
  "Decode token IDs, suppressing control tokens and joining byte fallbacks."
  [tokenizer token-ids]
  (let [out (StringBuilder.) bytes (ByteArrayOutputStream.)
        flush! (fn []
                 (when (pos? (.size bytes))
                   (.append out (String. (.toByteArray bytes) StandardCharsets/UTF_8))
                   (.reset bytes)))]
    (doseq [id token-ids]
      (let [token (get (:tokens tokenizer) id)
            type (get (:types tokenizer) id)]
        (when token
          (if-let [[_ hex] (re-matches #"<0x([0-9A-Fa-f]{2})>" token)]
            (.write bytes (Integer/parseInt hex 16))
            (do (flush!)
                (when-not (contains? #{3 5} type) (.append out token)))))))
    (flush!)
    (-> (str out) (str/replace "▁" " ") (str/replace-first #"^ " ""))))
