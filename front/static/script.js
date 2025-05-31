// static/script.js
const FRONTEND_API = "http://localhost:3000";  // 前端 API
const BACKEND_API = "http://localhost:8000";   // 後端 API

// 保存當前文章的句子和分數
let currentArticle = {
  sentences: [], // 所有句子的扁平列表
  scores: [],    // 對應扁平列表中每個句子的分數
  title: "",
  paragraphs: [] // 用於存儲段落結構，每個元素是一個段落(句子數組)
};

// 分句函數
function segmentText(text) {
  if (typeof text !== 'string') {
    console.error('segmentText: input text is not a string:', text);
    return { sentences: [], paragraphs: [] }; // 返回空結構以避免後續錯誤
  }
  // 1. 按換行符分割成段落 (處理多個換行符)
  const paragraphStrings = text.split(/\n+/).filter(p => p.trim().length > 0);
  
  let allSentences = [];
  let structuredParagraphs = [];
  
  // 2. 處理每個段落，將其分割成句子
  paragraphStrings.forEach(paraStr => {
    const sentencesInPara = paraStr
      .split(/([，。！？,.?!])/g) // Expanded regex to include English punctuation
      .reduce((acc, part) => {
        if (acc.length === 0) {
          acc.push(part);
        } else if (/[，。！？,.?!]/.test(part)) { // Expanded regex test here as well
          acc[acc.length - 1] += part; // 追加到上一個元素
        } else if (part.trim().length > 0){
          acc.push(part); // 否則，作為新元素添加
        }
        return acc;
      }, [])
      .filter(sentence => sentence && sentence.trim().length > 0);
    
    if (sentencesInPara.length > 0) {
      structuredParagraphs.push(sentencesInPara);
      allSentences = allSentences.concat(sentencesInPara);
    }
  });

  return {
    sentences: allSentences,          // 所有句子的扁平列表
    paragraphs: structuredParagraphs  // 段落結構 (句子數組的數組)
  };
}

// 自動高亮處理 (此函數不直接處理段落，它依賴 allSentences)
async function autoHighlight(sentences) {
  const response = await fetch(`${BACKEND_API}/api/auto-highlight`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ sentences })
  });

  if (!response.ok) {
    throw new Error('Auto-highlight request failed');
  }
  return await response.json();
}

// ==========================
// 主要的文章獲取和處理邏輯
// ==========================
async function loadAndProcessArticleFromUrl(url) {
  if (!url) {
    alert("Please paste the URL!");
    throw new Error("URL is empty");
  }

  const resultDiv = document.getElementById("result");
  const densitySlider = document.getElementById("highlight-density");
  resultDiv.innerHTML = '<div class="article-container"><div class="article-header"><p class="article-title">Loading...</p></div></div>';

  try {
    const response = await fetch(`${BACKEND_API}/api/extract-url`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify({ url })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server error: ${response.status} ${errorText}`);
    }

    const apiData = await response.json();
    if (!apiData.success || !apiData.data || typeof apiData.data.text !== 'string') {
      throw new Error("無法從API獲取有效的文章內容，或者文章文本缺失。");
    }

    const { sentences, paragraphs } = segmentText(apiData.data.text);

    if (sentences.length === 0) {
      currentArticle = { sentences: [], paragraphs: [], scores: [], title: apiData.data.title || "無標題" };
      updateHighlights(parseFloat(densitySlider.value) || 0.5); // 顯示空內容或提示
      alert("無法將文章內容分割成句子，或文章無有效文本內容。");
      throw new Error("文章無有效句子可供處理。");
    }

    // LOAD 按鈕功能：載入文章後立即自動高亮
    console.log("LOAD: Fetching initial highlights from model...");
    const highlightData = await autoHighlight(sentences);
    
    currentArticle = {
      sentences: sentences,
      paragraphs: paragraphs,
      scores: highlightData.scores, // 使用模型返回的分數
      title: apiData.data.title || "無標題"
    };

    const suggestedThreshold = highlightData.threshold !== undefined ? highlightData.threshold : 0.3; // 提供預設值
    densitySlider.value = suggestedThreshold;

    console.log("LOAD - Scores from initial AUTO-HIGHLIGHT:", JSON.parse(JSON.stringify(currentArticle.scores)));
    console.log("LOAD - Threshold from initial AUTO-HIGHLIGHT:", suggestedThreshold);

    updateHighlights(suggestedThreshold);
    document.getElementById("article-url").value = "";

  } catch (e) {
    console.error("處理文章時發生錯誤 (loadAndProcessArticleFromUrl):", e);
    resultDiv.innerHTML = `
      <div class="article-container">
        <div class="article-header">
          <p class="article-title">錯誤</p>
          <button class="close-btn" onclick="closeArticle()">&times;</button>
        </div>
        <div class="article-content"><p class="error-message">${e.message}</p></div>
      </div>`;
    //  這裡不再向上拋出錯誤，因為submit按鈕的finally塊會處理按鈕狀態，錯誤已顯示
  }
}

// 更新文章高亮顯示 (關鍵修改：基於段落結構渲染)
function updateHighlights(density) {
  console.log("Updating highlights with density:", density); // Added for debugging
  console.log("Current article paragraphs:", JSON.parse(JSON.stringify(currentArticle.paragraphs))); // Added for debugging
  console.log("Current article scores:", JSON.parse(JSON.stringify(currentArticle.scores))); // Added for debugging
  console.log("Total sentences in currentArticle.sentences:", currentArticle.sentences.length); // Added for debugging
  console.log("Total scores in currentArticle.scores:", currentArticle.scores.length); // Added for debugging

  if (!currentArticle.paragraphs || currentArticle.paragraphs.length === 0) {
    return;
  }

  const resultDiv = document.getElementById("result");
  let sentenceGlobalIndex = 0; 

  const paragraphsHtml = currentArticle.paragraphs.map(paragraphSentences => {
    const sentencesHtml = paragraphSentences.map(sentenceText => {
      const score = currentArticle.scores[sentenceGlobalIndex] || 0; 
      const isHighlighted = score >= density; // Condition for highlighting
      const currentSentenceIndex = sentenceGlobalIndex; 
      sentenceGlobalIndex++; 

      // Use class for highlighting instead of inline style
      return `<span 
        class="sentence${isHighlighted ? ' sentence-highlighted' : ''}" 
        data-index="${currentSentenceIndex}"
      >${sentenceText}</span>`;
    }).join(" "); 
    return `<p class="paragraph">${sentencesHtml}</p>`;
  }).join("");

  resultDiv.innerHTML = `
    <div class="article-container">
      <div class="article-header">
        <h2 class="article-title">${currentArticle.title}</h2>
        <button class="close-btn" onclick="closeArticle()">&times;</button>
      </div>
      <div class="article-content">
        ${paragraphsHtml}
      </div>
    </div>
  `;
}

// 關閉文章
function closeArticle() {
  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = "";
  currentArticle = {
    sentences: [],
    scores: [],
    title: "",
    paragraphs: [] // 確保重置時也包含 paragraphs
  };
}

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
    const res = await fetch(`${FRONTEND_API}/login`, {
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
    const response = await fetch(`${FRONTEND_API}/register`, {
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

// 確保 DOM 載入完成後才綁定事件
document.addEventListener('DOMContentLoaded', function() {
  // 綁定 URL 輸入表單的提交事件
  const urlForm = document.getElementById('url-form');
  if (urlForm) {
    urlForm.addEventListener('submit', async function(e) {
      e.preventDefault();
  const url = document.getElementById("article-url").value.trim();
      const submitButton = document.getElementById('submit-url');
      
      if (!url) {
        alert("請輸入 URL");
        return;
      }

      submitButton.disabled = true;
      submitButton.textContent = 'LOADING...';
      try {
        await loadAndProcessArticleFromUrl(url);
      } catch (error) {
        // 錯誤已在 loadAndProcessArticleFromUrl 中處理並顯示
        console.log("表單提交處理時捕獲到錯誤，已顯示。")
      } finally {
        submitButton.disabled = false;
        submitButton.textContent = 'LOAD';
      }
    });
  }

  // 滑桿事件 (依賴扁平的 currentArticle.sentences 和 currentArticle.scores)
  const densitySlider = document.getElementById('highlight-density');
  if (densitySlider) {
      densitySlider.addEventListener('input', function(e) {
          if (currentArticle.sentences.length > 0) { // 只有在有句子時才更新
              updateHighlights(parseFloat(e.target.value));
          }
      });
  }

  // AUTO 按鈕事件
  const autoButton = document.getElementById('auto-highlight');
  if (autoButton) {
      autoButton.addEventListener('click', async function() {
          console.log("AUTO button clicked.");
          if (!currentArticle.sentences || currentArticle.sentences.length === 0) {
              alert('請先載入文章。');
              return;
          }

          const userId = localStorage.getItem('user_id');
          if (!userId) {
              alert('無法獲取使用者ID，請確保您已登入。');
              // Optionally, use a default threshold or prompt login
              // For now, we use the current slider value if no user ID
              const densitySlider = document.getElementById("highlight-density");
              updateHighlights(parseFloat(densitySlider.value)); 
              return;
          }

          autoButton.disabled = true;
          autoButton.textContent = "CALCULATING...";
          const densitySlider = document.getElementById("highlight-density");

          try {
              console.log(`AUTO Button: Fetching personalized threshold for user ${userId}...`);
              const response = await fetch(`${FRONTEND_API}/api/user-threshold/${userId}`); // Assuming FRONTEND_API is correct
              
              if (!response.ok) {
                  const errorData = await response.json();
                  throw new Error(errorData.detail || '無法獲取個人化閾值。');
              }
              
              const data = await response.json();
              const personalizedThreshold = data.threshold;
              
              console.log("AUTO Button - Personalized threshold received:", personalizedThreshold);
              
              densitySlider.value = personalizedThreshold;
              updateHighlights(personalizedThreshold);
              
              // Show a notification for personalized threshold
              const feedbackNotification = document.getElementById('feedback-notification'); // Reuse existing notification element
              if (feedbackNotification) {
                feedbackNotification.textContent = 'Successfully changed!!'; // Updated success message
                const btnRect = autoButton.getBoundingClientRect();
                feedbackNotification.style.left = (window.scrollX + btnRect.left) + 'px'; 
                feedbackNotification.style.top = (window.scrollY + btnRect.bottom + 5) + 'px';
                feedbackNotification.style.display = 'block';
                setTimeout(() => { feedbackNotification.style.opacity = '1'; }, 10);
                
                clearTimeout(window.autoNotificationTimeoutId); 
                window.autoNotificationTimeoutId = setTimeout(() => {
                  feedbackNotification.style.opacity = '0';
                  setTimeout(() => { 
                    feedbackNotification.style.display = 'none';
                    feedbackNotification.textContent = 'Thx for ur feedback!'; // Reset text
                  }, 500); 
                }, 2500); // Show for 2.5 seconds
              }

          } catch (error) {
              console.error('Error fetching/applying personalized threshold (AUTO Button):', error);
              alert(`自動調整閾值失敗：${error.message}`);
              // Fallback to current slider value or default if error occurs
              updateHighlights(parseFloat(densitySlider.value)); 
          } finally {
              autoButton.disabled = false;
              autoButton.textContent = "AUTO";
          }
      });
  }
  
  // ... (其他如登出按鈕的初始化代碼) ...
  const logoutBtn = document.querySelector('.logout-btn'); // 假設登出按鈕有這個class
  if(logoutBtn) {
    logoutBtn.addEventListener('click', logout);
  }

  // Create and append the like/dislike buttons container
  const feedbackButtonsContainer = document.createElement('div');
  feedbackButtonsContainer.id = 'feedback-buttons-container';
  feedbackButtonsContainer.style.position = 'absolute'; // For positioning near mouse
  feedbackButtonsContainer.style.display = 'none'; // Hidden by default
  feedbackButtonsContainer.style.zIndex = '1000'; // Ensure it's on top
  feedbackButtonsContainer.style.background = 'white';
  feedbackButtonsContainer.style.border = '1px solid #ccc';
  feedbackButtonsContainer.style.padding = '5px';
  feedbackButtonsContainer.style.borderRadius = '5px';
  feedbackButtonsContainer.innerHTML = `
    <button id="like-btn" style="margin-right: 5px; cursor: pointer;">👍</button>
    <button id="dislike-btn" style="cursor: pointer;">👎</button>
  `;
  document.body.appendChild(feedbackButtonsContainer);

  // Create and append the feedback notification element
  const feedbackNotification = document.createElement('div');
  feedbackNotification.id = 'feedback-notification';
  feedbackNotification.style.position = 'absolute';
  feedbackNotification.style.display = 'none'; // Hidden by default
  feedbackNotification.style.opacity = '0'; // Start fully transparent for fade-in
  feedbackNotification.style.transition = 'opacity 0.5s ease-in-out'; // Fade transition
  feedbackNotification.style.zIndex = '1001'; 
  feedbackNotification.style.background = '#D4EDDA'; // Lighter green background
  feedbackNotification.style.color = '#155724'; // Darker text for contrast
  feedbackNotification.style.padding = '8px 12px';
  feedbackNotification.style.borderRadius = '8px'; // Slightly more rounded corners
  feedbackNotification.style.fontSize = '0.9em';
  feedbackNotification.style.boxShadow = '0 2px 5px rgba(0,0,0,0.1)'; // Optional: subtle shadow
  feedbackNotification.textContent = 'Thx for ur feedback!';
  document.body.appendChild(feedbackNotification);

  let notificationTimeoutId = null; 
  let hideButtonsTimeoutId = null; // Timer for delaying button hide

  // Event delegation for highlighted sentences
  const resultDiv = document.getElementById('result');
  if (resultDiv) {
    resultDiv.addEventListener('mouseover', function(event) {
      const targetSentence = event.target.closest('.sentence'); 
      if (targetSentence) {
        clearTimeout(hideButtonsTimeoutId); // Clear any pending hide
        feedbackButtonsContainer.style.left = (event.pageX + 10) + 'px';
        feedbackButtonsContainer.style.top = (event.pageY + 5) + 'px';
        feedbackButtonsContainer.style.display = 'flex';
        feedbackButtonsContainer.setAttribute('data-sentence-index', targetSentence.dataset.index);
      } else {
        // If not over a sentence, start timer to hide buttons (unless over buttons themselves)
        if (!feedbackButtonsContainer.contains(event.relatedTarget)) {
            clearTimeout(hideButtonsTimeoutId);
            hideButtonsTimeoutId = setTimeout(() => {
                feedbackButtonsContainer.style.display = 'none';
            }, 400); // Hide after 400ms
        }
      }
    });

    // When mouse leaves a sentence or the general result area
    resultDiv.addEventListener('mouseout', function(event) {
        // Check if the mouse is moving to the buttons container or outside relevant areas
        if (!event.relatedTarget || 
            (!event.relatedTarget.closest('.sentence') && !feedbackButtonsContainer.contains(event.relatedTarget))) {
            clearTimeout(hideButtonsTimeoutId);
            hideButtonsTimeoutId = setTimeout(() => {
                // Check again if mouse is now over the buttons before hiding
                if (!feedbackButtonsContainer.matches(':hover')) { 
                    feedbackButtonsContainer.style.display = 'none';
                }
            }, 400); 
        }
    });

    // Keep buttons visible if mouse enters the button container itself
    feedbackButtonsContainer.addEventListener('mouseover', function() {
        clearTimeout(hideButtonsTimeoutId);
    });

    // When mouse leaves the button container itself, start timer to hide it
    feedbackButtonsContainer.addEventListener('mouseleave', function() {
        clearTimeout(hideButtonsTimeoutId);
        hideButtonsTimeoutId = setTimeout(() => {
            feedbackButtonsContainer.style.display = 'none';
        }, 400);
    });

    // Add click listeners for the buttons
    const likeBtn = document.getElementById('like-btn');
    const dislikeBtn = document.getElementById('dislike-btn');

    if (likeBtn) {
      likeBtn.addEventListener('click', async function(event) {
        const sentenceIndex = parseInt(feedbackButtonsContainer.getAttribute('data-sentence-index'), 10);
        const currentScore = currentArticle.scores[sentenceIndex];
        console.log(`Liked sentence index: ${sentenceIndex}, Score: ${currentScore}`);
        
        // Send feedback to backend
        try {
          const userId = localStorage.getItem('user_id');
          if (!userId) {
            console.error("User ID not found for feedback.");
            return;
          }
          const response = await fetch(`${FRONTEND_API}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              user_id: parseInt(userId, 10),
              label: 1,
              sentence_score: currentScore 
            })
          });
          const result = await response.json();
          if (!result.success) {
            console.error("Failed to save like feedback:", result.detail);
          }
        } catch (error) {
          console.error("Error sending like feedback:", error);
        }

        const btnRect = likeBtn.getBoundingClientRect();
        feedbackNotification.style.left = (window.scrollX + btnRect.left) + 'px'; 
        feedbackNotification.style.top = (window.scrollY + btnRect.bottom + 5) + 'px';
        feedbackNotification.style.display = 'block';
        setTimeout(() => { feedbackNotification.style.opacity = '1'; }, 10); // Fade in shortly after display block

        feedbackButtonsContainer.style.display = 'none'; 
        
        clearTimeout(notificationTimeoutId); // Clear any existing timeout
        notificationTimeoutId = setTimeout(() => {
          feedbackNotification.style.opacity = '0'; // Start fade out
          // Wait for fade out to complete before setting display to none
          setTimeout(() => { feedbackNotification.style.display = 'none'; }, 500); // Corresponds to transition duration
        }, 1000); // Start fade out after 1 second
      });
    }

    if (dislikeBtn) {
      dislikeBtn.addEventListener('click', async function(event) {
        const sentenceIndex = parseInt(feedbackButtonsContainer.getAttribute('data-sentence-index'), 10);
        const currentScore = currentArticle.scores[sentenceIndex];
        console.log(`Disliked sentence index: ${sentenceIndex}, Score: ${currentScore}`);

        // Send feedback to backend
        try {
          const userId = localStorage.getItem('user_id');
          if (!userId) {
            console.error("User ID not found for feedback.");
            return;
          }
          const response = await fetch(`${FRONTEND_API}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              user_id: parseInt(userId, 10),
              label: 0,
              sentence_score: currentScore 
            })
          });
          const result = await response.json();
          if (!result.success) {
            console.error("Failed to save dislike feedback:", result.detail);
          }
        } catch (error) {
          console.error("Error sending dislike feedback:", error);
        }

        const btnRect = dislikeBtn.getBoundingClientRect();
        feedbackNotification.style.left = (window.scrollX + btnRect.left) + 'px';
        feedbackNotification.style.top = (window.scrollY + btnRect.bottom + 5) + 'px';
        feedbackNotification.style.display = 'block';
        setTimeout(() => { feedbackNotification.style.opacity = '1'; }, 10); // Fade in shortly after display block

        feedbackButtonsContainer.style.display = 'none';

        clearTimeout(notificationTimeoutId); // Clear any existing timeout
        notificationTimeoutId = setTimeout(() => {
          feedbackNotification.style.opacity = '0'; // Start fade out
          // Wait for fade out to complete before setting display to none
          setTimeout(() => { feedbackNotification.style.display = 'none'; }, 500); // Corresponds to transition duration
        }, 1000); // Start fade out after 1 second
      });
    }
  }

});

// 更新 CSS 樣式 (確保 .paragraph 有樣式)
const styleElement = document.getElementById('dynamic-styles') || document.createElement('style');
styleElement.id = 'dynamic-styles';
styleElement.textContent = `
  .article-content {
    line-height: 1.8;
    font-size: 16px;
    text-align: left; /* 確保文本左對齊 */
  }
  .paragraph {
    margin-bottom: 1em; /* 段落下邊距 */
  }
  .sentence {
    display: inline; /* 句子內聯顯示 */
    padding: 0 1px; /* 上下0, 左右1px padding */
    transition: background-color 0.3s; /* 保留背景色變化的過渡效果 */
    margin-right: 2px; /* 句子間的微小間隔，如果需要 */
    box-decoration-break: clone; /* 確保多行句子的背景也能正确应用padding */
    -webkit-box-decoration-break: clone; /* 兼容 WebKit */
  }
  .sentence-highlighted {
    background-color: rgba(255, 250, 150, 0.5); /* Normal highlight color */
  }
  .sentence-highlighted:hover {
    background-color: rgba(255, 240, 100, 0.7); /* Darker highlight on hover */
  }
  /* .sentence:hover 規則被移除，以防止鼠標懸停時改變背景 */
`;
if (!document.getElementById('dynamic-styles')) {
  document.head.appendChild(styleElement);
}
