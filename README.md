# vllm-clj — vLLM 推論リクエストを EDN データとして

[![CI](https://github.com/kotoba-lang/vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/kotoba-lang/vllm/actions/workflows/ci.yml)

An LLM inference request against a **vLLM (OpenAI-compatible) server**, defined
as **plain EDN data**, with a pure renderer to the wire shape and a normalized
response parser. Like its sibling libraries, vllm-clj treats the request as data
— generated, diffed, versioned, logged, and stored (Datomic / kotoba) like any
other EDN value — and leaves the socket to the host.

- **Zero third-party runtime deps.** Request/wire namespaces remain portable
  `.cljc` (JVM, ClojureScript, SCI); the local GGUF reader is JVM-specific.
- **Don't build JSON text.** vllm-clj renders requests to *string-keyed Clojure
  maps* and parses *string-keyed* responses; the host transport does the actual
  HTTP and JSON (de)serialization (clj-http, http-kit, hato, `js/fetch`, …).
- **Data-first.** The request is plain EDN; the network is a host-injected
  `vllm.ports/ITransport`, not a baked-in client.

Sibling libs: [jsonlogic-clj](../jsonlogic-clj) (rules-as-data),
[mcp-clj](../mcp-clj) (protocol/transport-as-data),
[torch-clj](../torch-clj) (module-graph-as-data).

## Why a shared library (org placement)

Per the three-org rule, the **reusable** request/wire kernel lives in
**com-junkawasaki**; **public-benefit** actor instances that call models live in
**etzhayyim**; any **business/private deployment** lives in **gftdcojp**.
vllm-clj is the dep — it carries no prompts and no HTTP client (those are
host-injected ports).

## The request: an inference call as EDN (`vllm.request`)

Two endpoints — chat (`/v1/chat/completions`) and text completion
(`/v1/completions`) — built with threadable data:

```clojure
(require '[vllm.request :as req])

(-> (req/chat "meta-llama/Llama-3-8B-Instruct"
              [(req/system "You are concise.")
               (req/user   "Explain CACAO in one sentence.")])
    (req/with-sampling {:vllm/temperature 0.2 :vllm/max-tokens 256})
    (req/with-base-url "http://gpu-node:8000"))
;; => {:vllm/endpoint :chat :vllm/model "…"
;;     :vllm/messages [{:vllm/role :system :vllm/content "…"} …]
;;     :vllm/sampling {:vllm/temperature 0.2 :vllm/max-tokens 256 …}
;;     :vllm/base-url "http://gpu-node:8000"}
```

## Sampling (`vllm.sampling`)

The full OpenAI surface plus vLLM extensions (`top_k`, `min_p`,
`repetition_penalty`, `best_of`, and the `guided_*` structured-output knobs),
each a namespaced key. `to-wire` maps keys to wire fields (unknown keys dropped),
`problems` does pure range checks:

```clojure
(require '[vllm.sampling :as sampling])

(sampling/to-wire {:vllm/temperature 0.7 :vllm/top-k 40 :vllm/max-tokens 128})
;; => {"temperature" 0.7 "top_k" 40 "max_tokens" 128}

(sampling/problems {:vllm/temperature 9.0})
;; => [{:vllm/severity :error :vllm/code :sampling/range …}]
```

## Wire (`vllm.wire`)

`render` produces the string-keyed body; `parse-response` / `parse-chunk`
normalize server JSON (already parsed by the host) back into `:vllm/*` EDN —
including SSE streaming deltas:

```clojure
(require '[vllm.wire :as wire])

(wire/render (req/chat "m" [(req/user "hi")]))
;; => {"model" "m" "temperature" 0.7 "top_p" 1.0 "max_tokens" 512
;;     "messages" [{"role" "user" "content" "hi"}]}

(wire/parse-response
  {"choices" [{"message" {"content" "Hello!"} "finish_reason" "stop"}]
   "usage" {"total_tokens" 10}})
;; => {:vllm/text "Hello!" :vllm/choices [{:vllm/finish-reason :stop …}]
;;     :vllm/usage {:vllm/total-tokens 10}}
```

## Validation (`vllm.validate`)

Pure structural checks (model present, roles valid, content strings, endpoint
known) plus the sampling range checks, as a vector of problem maps. `valid?` is
true iff there are no `:error`s.

```clojure
(require '[vllm.validate :as v])
(v/valid? (req/chat "m" [(req/user "hi")]))   ;=> true
(v/valid? (req/chat ""  []))                   ;=> false (no model, no messages)
```

## Ports & the call (`vllm.ports`, `vllm.core`)

The host injects an `ITransport` (and optionally `IStreamTransport`).
`vllm.core/complete` validates, builds the request spec, sends, and normalizes —
never touching a socket itself:

```clojure
(require '[vllm.ports :as ports] '[vllm.core :as core])

;; a host transport built on its HTTP client of choice
(def transport
  (ports/fn-transport
    (fn [{:keys [url method headers body]}]
      ;; host: JSON-encode body, POST, JSON-decode response
      {:status 200 :body (http-post! url headers body)})))

(core/chat transport "meta-llama/Llama-3-8B-Instruct"
           [(req/user "Hello!")])
;; => {:vllm/text "Hi there!" :vllm/usage {…} …}

;; streaming token deltas over an IStreamTransport
(core/stream transport (req/chat "m" [(req/user "count to 3")])
             (fn [delta] (print delta)))
```

Validation short-circuits before the wire (`{:vllm/error :validation …}`); a
non-2xx response becomes `{:vllm/error :http :vllm/status …}`. With the default
`ports/no-transport`, vllm-clj does no I/O and `complete` throws — by design.

## Embedded GGUF foundation (`vllm.gguf`)

The JVM runtime now has a validated GGUF v3 reader as the first local-inference
layer. `open-file` parses scalar, string, nested-array metadata and the tensor
catalog, validates alignment/block sizes/file windows, and leaves model payloads
on disk. `read-tensor-bytes` reads only one requested tensor window. Catalog
support covers F32/F16/BF16, integer tensors, and the common Q4/Q5/Q8/K-block
encodings; dequantization and Transformer execution are the next layer.

```clojure
(require '[vllm.gguf :as gguf])

(with-open [model (gguf/open-file "/models/model.gguf")]
  {:architecture (get (:metadata model) "general.architecture")
   :tensors (gguf/tensor-names model)
   :token-embedding (gguf/tensor-info model "token_embd.weight")})
```

This does not yet make vllm-clj an Ollama replacement: tokenizer execution,
quantized matrix kernels, Transformer/KV-cache generation, model lifecycle,
and the Ollama HTTP surface remain explicit required work.

## Test

```
clojure -M:test
```
