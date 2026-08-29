(ns vllm.ollama-server
  "JDK HttpServer adapter exposing the embedded Ollama handler as JSON/NDJSON."
  (:require [json.data-json :as json]
            [clojure.string :as str]
            [vllm.ollama :as ollama])
  (:import [com.sun.net.httpserver HttpExchange HttpHandler HttpServer]
           [java.io Closeable]
           [java.net InetSocketAddress]
           [java.nio.charset StandardCharsets]
           [java.util.concurrent Executors]))

(defrecord Server [^HttpServer server executor port]
  Closeable
  (close [_]
    (.stop server 0)
    (.shutdownNow ^java.util.concurrent.ExecutorService executor)))

(defn- utf8-bytes [value] (.getBytes (str value) StandardCharsets/UTF_8))

(defn- read-body [^HttpExchange exchange]
  (let [payload (.readAllBytes (.getRequestBody exchange))]
    (if (zero? (alength payload)) {}
        (json/read-str (String. payload StandardCharsets/UTF_8)))))

(defn- send! [^HttpExchange exchange status content-type body]
  (let [payload (utf8-bytes (json/write-str body))]
    (.set (.getResponseHeaders exchange) "Content-Type" content-type)
    (.sendResponseHeaders exchange status (alength payload))
    (with-open [out (.getResponseBody exchange)] (.write out payload))))

(defn- handler [runtime]
  (reify HttpHandler
    (handle [_ exchange]
      (try
        (let [method (keyword (str/lower-case (.getRequestMethod exchange)))
              path (.getPath (.getRequestURI exchange))
              body (read-body exchange)
              streaming? (and (contains? #{"/api/generate" "/api/chat"} path)
                              (contains? @(:models runtime) (get body "model"))
                              (not (false? (get body "stream" true))))]
          (if streaming?
            (do
              (.set (.getResponseHeaders exchange) "Content-Type" "application/x-ndjson")
              (.sendResponseHeaders exchange 200 0)
              (with-open [out (.getResponseBody exchange)]
                (ollama/handle runtime
                               {:method method :path path :body body
                                :emit (fn [chunk]
                                        (.write out (utf8-bytes (str (json/write-str chunk) "\n")))
                                        (.flush out))})))
            (let [{:keys [status body]} (ollama/handle runtime
                                                       {:method method :path path :body body})]
              (send! exchange status "application/json" body))))
        (catch Throwable error
          (try (send! exchange 500 "application/json" {"error" (.getMessage error)})
               (catch Throwable _)))))))

(defn start!
  "Start an Ollama-compatible HTTP server. `port` may be 0 for an ephemeral
  test port. Returns a Closeable Server with its selected `:port`."
  ([runtime] (start! runtime {:host "127.0.0.1" :port 11434}))
  ([runtime {:keys [host port backlog threads]
             :or {host "127.0.0.1" port 11434 backlog 128 threads 4}}]
   (let [server (HttpServer/create (InetSocketAddress. host (int port)) (int backlog))
         executor (Executors/newFixedThreadPool (int threads))]
     (.createContext server "/" (handler runtime))
     (.setExecutor server executor)
     (.start server)
     (->Server server executor (.getPort (.getAddress server))))))
