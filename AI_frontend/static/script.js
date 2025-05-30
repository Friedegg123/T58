// static/script.js
const API = "http://localhost:8000";

// ==========================
// 登入功能：POST /login
// ==========================
async function login() {
  const username = document.getElementById("login-username").value.trim();
  const password = document.getElementById("login-password").value.trim();
  const msgElem = document.getElementById("login-error");

  msgElem.textContent = "";

  if (!username || !password) {
    msgElem.textContent = "請輸入帳號與密碼";
    return;
  }

  const formData = new FormData();
  formData.append("username", username);
  formData.append("password", password);

  try {
    const res = await fetch(`${API}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },  
      body: JSON.stringify({ username, password })
    });

    const data = await res.json();
    if (res.ok && data.success) {
      localStorage.setItem("user_id", data.user_id);
      localStorage.setItem("username", data.username);
      window.location.href = "highlight.html";
    } else {
      msgElem.textContent = data.detail || data.message || "登入失敗";
    }
  } catch (err) {
    msgElem.textContent = "無法連線到伺服器";
  }
}


// ==========================
// 註冊功能：POST /register
// ==========================
async function register() {
  const username = document.getElementById("reg-username").value.trim();
  const password = document.getElementById("reg-password").value.trim();
  const msgElem = document.getElementById("reg-msg");

  msgElem.textContent = "";

  const formData = new FormData();
  formData.append("username", username);
  formData.append("password", password);

  try {
    const response = await fetch(`${API}/register`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (data.success) {
      msgElem.innerText = "註冊成功，請返回登入頁";
      msgElem.style.color = "green";
      setTimeout(() => (window.location.href = "index.html"), 1000);
    } else {
      msgElem.innerText = data.message;
      msgElem.style.color = "red";
    }
  } catch (error) {
    msgElem.innerText = "無法連線到伺服器";
    msgElem.style.color = "red";
  }
}

// ==========================
// 登出功能
// ==========================
function logout() {
  localStorage.removeItem("token");
  localStorage.removeItem("user_id");
  window.location.href = "index.html";
}

// ==========================
// 顯示高亮句子（目前是假資料）
// ==========================
async function fetchArticle() {
  const url = document.getElementById("article-url").value.trim();
  if (!url) return alert("請貼上網址！");

  try {
    const res = await fetch("/highlight-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await res.json();              // { sentences:[], scores:[] }
    highlights.length = 0;                      // 清空舊資料
    data.sentences.forEach((s, i) =>
      highlights.push({ text: s, score: data.scores[i] })
    );
    render(parseFloat(threshold.value));        // 重新渲染
  } catch (e) {
    alert(e.message);
  }
}

// ==========================
// 回饋（目前模擬）
// ==========================


async function sendFeedback(sentence, label) {
  const user_id = localStorage.getItem("user_id");

  try {
    await fetch(`${API}/feedback`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ sentence, label, user_id })
    });
  } catch (e) {
    console.log("回饋送出失敗");
  }

  // 顯示提示
  const toast = document.createElement('div');
  toast.textContent = "已收到回饋，謝謝你！";
  toast.style.position = 'fixed';
  toast.style.bottom = '20px';
  toast.style.right = '20px';
  toast.style.background = '#4ade80';
  toast.style.color = '#fff';
  toast.style.padding = '10px 16px';
  toast.style.borderRadius = '10px';
  toast.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
  toast.style.zIndex = 9999;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2000);
}

document.getElementById("threshold").addEventListener("input", e => {
  document.getElementById("threshold-value").textContent = parseFloat(e.target.value).toFixed(2);
  fetchHighlights(); // 重新篩選
});

document.getElementById("theme-toggle").addEventListener("click", () => {
  document.body.classList.toggle("dark");
});
