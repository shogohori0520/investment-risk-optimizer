/* ===== Theme Toggle ===== */
function initTheme() {
  const saved = localStorage.getItem('book-theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
}
function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('book-theme', next);
}
initTheme();

/* ===== Sidebar Toggle (Mobile) ===== */
function toggleSidebar() {
  document.querySelector('.sidebar').classList.toggle('show');
}
document.addEventListener('click', (e) => {
  const sidebar = document.querySelector('.sidebar');
  if (sidebar && sidebar.classList.contains('show') &&
      !sidebar.contains(e.target) && !e.target.closest('.menu-toggle')) {
    sidebar.classList.remove('show');
  }
});

/* ===== Active Section Tracking ===== */
function initScrollSpy() {
  const headings = document.querySelectorAll('.content-area h1[id], .content-area h2[id], .content-area h3[id]');
  const navLinks = document.querySelectorAll('.sidebar-nav a');
  if (!headings.length || !navLinks.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(l => l.classList.remove('active'));
        const link = document.querySelector(`.sidebar-nav a[href="#${entry.target.id}"]`);
        if (link) link.classList.add('active');
      }
    });
  }, { rootMargin: '-80px 0px -60% 0px' });

  headings.forEach(h => observer.observe(h));
}

/* ===== Reading Progress ===== */
function initProgress() {
  const bar = document.querySelector('.chapter-progress .bar');
  if (!bar) return;
  window.addEventListener('scroll', () => {
    const docH = document.documentElement.scrollHeight - window.innerHeight;
    const pct = docH > 0 ? (window.scrollY / docH) * 100 : 0;
    bar.style.height = pct + '%';
  });
}

/* ===== Compound Interest Calculator ===== */
function calcCompound() {
  const principal = parseFloat(document.getElementById('calc-principal')?.value) || 0;
  const monthly = parseFloat(document.getElementById('calc-monthly')?.value) || 0;
  const rate = parseFloat(document.getElementById('calc-rate')?.value) || 0;
  const years = parseFloat(document.getElementById('calc-years')?.value) || 0;
  const r = rate / 100 / 12;
  const n = years * 12;

  let fv;
  if (r === 0) {
    fv = principal + monthly * n;
  } else {
    fv = principal * Math.pow(1 + r, n) + monthly * ((Math.pow(1 + r, n) - 1) / r);
  }
  const totalInvested = principal + monthly * n;
  const profit = fv - totalInvested;

  const el = document.getElementById('calc-compound-result');
  if (el) {
    el.innerHTML = `
      <div>最終資産額: <strong>¥${Math.round(fv).toLocaleString()}</strong></div>
      <div style="font-size:0.9rem;margin-top:0.3rem;">
        投資元本: ¥${Math.round(totalInvested).toLocaleString()} ／
        運用益: ¥${Math.round(profit).toLocaleString()}
        （利益率: ${totalInvested > 0 ? ((profit / totalInvested) * 100).toFixed(1) : 0}%）
      </div>`;
  }
}

/* ===== Required Savings Calculator ===== */
function calcRequired() {
  const target = parseFloat(document.getElementById('calc-target')?.value) || 0;
  const rate = parseFloat(document.getElementById('calc-req-rate')?.value) || 0;
  const years = parseFloat(document.getElementById('calc-req-years')?.value) || 0;
  const r = rate / 100 / 12;
  const n = years * 12;

  let monthly;
  if (r === 0) {
    monthly = n > 0 ? target / n : 0;
  } else {
    monthly = target * r / (Math.pow(1 + r, n) - 1);
  }

  const el = document.getElementById('calc-required-result');
  if (el) {
    el.innerHTML = `
      <div>必要な毎月の積立額: <strong>¥${Math.round(monthly).toLocaleString()}</strong></div>
      <div style="font-size:0.9rem;margin-top:0.3rem;">
        総投資額: ¥${Math.round(monthly * n).toLocaleString()} ／
        運用益: ¥${Math.round(target - monthly * n).toLocaleString()}
      </div>`;
  }
}

/* ===== Quiz ===== */
function checkQuiz(quizId) {
  const quiz = document.getElementById(quizId);
  if (!quiz) return;
  const selected = quiz.querySelector('input[type="radio"]:checked');
  const feedbackCorrect = quiz.querySelector('.quiz-feedback.correct');
  const feedbackWrong = quiz.querySelector('.quiz-feedback.wrong');

  if (!selected) return;

  feedbackCorrect.style.display = 'none';
  feedbackWrong.style.display = 'none';

  if (selected.dataset.correct === 'true') {
    feedbackCorrect.style.display = 'block';
  } else {
    feedbackWrong.style.display = 'block';
  }
}

/* ===== Font Size ===== */
let fontSize = parseFloat(localStorage.getItem('book-fontsize') || '16');
function changeFontSize(delta) {
  fontSize = Math.max(12, Math.min(22, fontSize + delta));
  document.documentElement.style.fontSize = fontSize + 'px';
  localStorage.setItem('book-fontsize', fontSize);
}
document.documentElement.style.fontSize = fontSize + 'px';

/* ===== Init ===== */
document.addEventListener('DOMContentLoaded', () => {
  initScrollSpy();
  initProgress();
});
