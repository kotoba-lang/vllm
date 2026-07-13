(ns vllm.metal-worker
  "Synchronous JVM client for num's persistent Deno WebGPU/Metal worker."
  (:require [clojure.data.json :as json]
            [vllm.accelerator :as accelerator])
  (:import [java.io BufferedReader BufferedWriter Closeable InputStreamReader OutputStreamWriter]
           [java.nio ByteBuffer]
           [java.util Base64 UUID]
           [java.lang ProcessBuilder$Redirect]))

(defn- fail [message data] (throw (ex-info (str "vllm.metal-worker: " message) data)))

(defn- buffer-bytes ^bytes [^ByteBuffer source]
  (let [copy (.duplicate source) result (byte-array (.remaining copy))]
    (.get copy result)
    result))

(defn- exchange! [{:keys [^BufferedWriter writer ^BufferedReader reader]} request]
  (.write writer (str (json/write-str request) "\n"))
  (.flush writer)
  (let [line (.readLine reader)]
    (when-not line (fail "worker closed its output" {:request (:op request)}))
    (let [response (json/read-str line :key-fn keyword)]
      (when-not (:ok response)
        (fail "worker rejected request" {:request (:op request) :response response}))
      response)))

(defrecord MetalWorker [^Process process ^BufferedWriter writer ^BufferedReader reader lock]
  accelerator/IMatrixAccelerator
  (upload-q8! [this id rows columns bytes]
    (locking lock
      (let [handle (str id "-" (UUID/randomUUID))
            encoded (.encodeToString (Base64/getEncoder)
                                     (if (instance? ByteBuffer bytes)
                                       (buffer-bytes bytes)
                                       ^bytes bytes))]
        (exchange! this {:op "upload-q8" :id handle :rows rows :cols columns
                         :data encoded})
        handle)))
  (gemv! [this handle x]
    (locking lock
      (float-array (:y (exchange! this {:op "gemv" :id handle :x (vec x)})))))
  (release! [this handle]
    (locking lock (exchange! this {:op "release" :id handle}) nil))
  accelerator/IBatchedMatrixAccelerator
  (gemv-many! [this requests]
    (locking lock
      (mapv (comp float-array :y)
            (:results
             (exchange! this
                        {:op "gemv-many"
                         :requests (mapv (fn [[handle x]]
                                           {:id handle :x (vec x)})
                                         requests)})))))
  accelerator/IQuantizedMatrixAccelerator
  (upload-quantized! [this type id rows columns bytes]
    (when-not (contains? #{:q8-0 :q4-0} type)
      (fail "unsupported Metal quantization" {:type type}))
    (locking lock
      (let [handle (str id "-" (UUID/randomUUID))
            encoded (.encodeToString (Base64/getEncoder)
                                     (if (instance? ByteBuffer bytes)
                                       (buffer-bytes bytes)
                                       ^bytes bytes))]
        (exchange! this {:op (case type :q8-0 "upload-q8" :q4-0 "upload-q4")
                         :id handle :rows rows :cols columns :data encoded})
        handle)))
  Closeable
  (close [_]
    (try (.close writer) (catch Exception _))
    (try (.close reader) (catch Exception _))
    (.destroy process)))

(defn start
  "Start a persistent Metal worker. `command` defaults to the compiled num
  worker in a sibling checkout and may be overridden for packaging."
  ([] (start ["deno" "run" "--allow-all" "../num/target/deno-q8-worker.cjs"]))
  ([command]
   (let [builder (ProcessBuilder. ^java.util.List command)
         _ (.redirectError builder ProcessBuilder$Redirect/INHERIT)
         process (.start builder)]
     (->MetalWorker process
                    (BufferedWriter. (OutputStreamWriter. (.getOutputStream process)))
                    (BufferedReader. (InputStreamReader. (.getInputStream process)))
                    (Object.)))))

(defn stats [worker]
  (locking (:lock worker) (exchange! worker {:op "stats"})))
