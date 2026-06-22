const form = document.querySelector("#askForm");
const questionInput = document.querySelector("#question");
const answerBox = document.querySelector("#answer");
const sourcesBox = document.querySelector("#sources");
const statusBox = document.querySelector("#status");
const indexButton = document.querySelector("#indexButton");

function setStatus(message, kind = "") {
  statusBox.textContent = message;
  statusBox.className = `status ${kind}`;
}

function renderSources(sources) {
  sourcesBox.innerHTML = "";
  if (!sources.length) return;

  const title = document.createElement("h2");
  title.textContent = "Nguồn tham khảo";
  sourcesBox.appendChild(title);

  for (const source of sources) {
    const item = document.createElement("div");
    item.className = "source";
    item.innerHTML = `
      <div class="source-meta">${source.source} · trang ${source.page} · score ${source.score.toFixed(3)}</div>
      <p>${source.text}</p>
    `;
    sourcesBox.appendChild(item);
  }
}

indexButton.addEventListener("click", async () => {
  setStatus("Đang build index...");
  answerBox.textContent = "";
  sourcesBox.innerHTML = "";

  const response = await fetch("/api/index", { method: "POST" });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.detail || "Không thể build index.", "error");
    return;
  }
  setStatus(`Đã index ${data.chunks} đoạn từ ${data.documents} tài liệu.`, "success");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  setStatus("Đang truy xuất tài liệu...");
  answerBox.textContent = "";
  sourcesBox.innerHTML = "";

  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.detail || "Không thể trả lời câu hỏi.", "error");
    return;
  }

  setStatus("Hoàn tất.", "success");
  answerBox.textContent = data.answer;
  renderSources(data.sources);
});
