import axios from "axios";
import { useState } from "react";
import "./App.css";

const axiosInstance = axios.create({
  baseURL: "http://localhost:5001",
});

function App() {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState("summary");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [question, setQuestion] = useState("");
  const [chatLog, setChatLog] = useState([]);
  const [abortController, setAbortController] = useState(null);

  const extractTopic = () => {
    if (text.trim()) {
      const firstSentence = text.split(/[.?!]/)[0];
      return firstSentence.length > 50 ? firstSentence.slice(0, 50) + "..." : firstSentence;
    } else if (file) {
      return file.name;
    }
    return "Unknown Topic";
  };

  const handleSubmit = async () => {
    setError("");
    setResult(null);
    setLoading(true);

    if (text.trim() && file) {
      setError("Please enter only text or upload a file, not both.");
      setLoading(false);
      return;
    }

    if (!text.trim() && !file) {
      setError("Please enter text or upload a file.");
      setLoading(false);
      return;
    }

    const formData = new FormData();
    if (file) formData.append("file", file);
    else formData.append("text", text);

    const endpoint = mode === "summary" ? "/summarize" : "/generate-questions";
    const controller = new AbortController();
    setAbortController(controller);

    try {
      const response = await axiosInstance.post(endpoint, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        signal: controller.signal,
      });

      const topic = extractTopic();
      const historyEntry = {
        mode,
        data: response.data,
        topic,
        timestamp: new Date().toLocaleString(),
      };

      setResult(response.data);
      setHistory((prev) => [historyEntry, ...prev]);
    } catch (err) {
      if (axios.isCancel(err) || err.name === "CanceledError") {
        console.log("Request aborted.");
      } else {
        setError("Failed to connect to backend. Is Flask running?");
      }
    } finally {
      setLoading(false);
      setAbortController(null);
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) return;
    const userMessage = { role: "user", content: question };
    setChatLog((prev) => [...prev, userMessage]);

    try {
      const response = await axiosInstance.post("/ask", { question });
      const assistantMessage = { role: "assistant", content: response.data.answer };
      setChatLog((prev) => [...prev, assistantMessage]);
      setQuestion("");
    } catch (err) {
      setError("Failed to get chatbot response.");
    }
  };

  const handleTextChange = (e) => {
    const newText = e.target.value;
    setText(newText);
    if (file) {
      setFile(null);
      const fileInput = document.getElementById("fileInput");
      if (fileInput) fileInput.value = "";
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (text) {
      setText("");
    }
  };

  const handleReset = () => {
    if (abortController) abortController.abort();
    setText("");
    setFile(null);
    setResult(null);
    setError("");
    setQuestion("");

    const fileInput = document.getElementById("fileInput");
    if (fileInput) fileInput.value = "";
  };

  const clearChatOnly = () => {
    setChatLog([]);
    setQuestion("");
  };

  return (
    <div className="container">
      <h2>AskLens</h2>

      <textarea
        rows="8"
        value={text}
        onChange={handleTextChange}
        disabled={loading}
        placeholder="Enter text or leave empty if uploading a file..."
      />

      <input
        id="fileInput"
        type="file"
        accept=".png,.jpg,.jpeg"
        onChange={handleFileChange}
        disabled={loading}
      />

      <div className="controls">
        <select value={mode} onChange={(e) => setMode(e.target.value)} disabled={loading}>
          <option value="summary">Summary</option>
          <option value="questions">Questions</option>
        </select>
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "Processing..." : "Submit"}
        </button>
        <button onClick={handleReset} style={{ marginLeft: "10px" }}>
          Reset
        </button>
      </div>

      {error && <div className="error">❌ {error}</div>}

      {result?.summary && (
        <div className="output">
          <h3>📄 Summary</h3>
          <p>{result.summary}</p>
        </div>
      )}

      {result?.questions && (
        <div className="output">
          <h3>❓ Generated Questions</h3>
          <ol>
            {result.questions.map((q, i) => (
              <li key={i}>
                {q.question}
                <br />
                <strong>Answer:</strong> {q.answer}
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="chatbot">
        <h3>💬 Ask Questions</h3>
        <div className="chat-log">
          {chatLog.map((msg, i) => (
            <div key={i} className={`chat-message ${msg.role}`}>
              <div className="chat-bubble">
                <strong>{msg.role === "user" ? "You" : "Bot"}:</strong> {msg.content}
              </div>
            </div>
          ))}
        </div>

        <div className="chat-input">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask anything..."
          />
          <button onClick={handleAsk}>Ask</button>
        </div>

        {chatLog.length > 0 && (
          <button onClick={clearChatOnly} className="clear-chat-btn">
            Clear Chat
          </button>
        )}
      </div>

      {history.length > 0 && (
        <div className="history">
          <h3>📚 History</h3>
          <ul>
            {history.map((item, i) => (
              <li key={i}>
                <strong>{item.mode === "summary" ? "Summary" : "Questions"}</strong> —{" "}
                <em>{item.topic}</em> <br />
                <small>🕒 {item.timestamp}</small>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
