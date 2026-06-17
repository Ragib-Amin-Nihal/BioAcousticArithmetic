/* Small UI helpers: mobile navbar burger + copy-to-clipboard for BibTeX. */
document.addEventListener("DOMContentLoaded", function () {
  // Bulma navbar burger
  document.querySelectorAll(".navbar-burger").forEach(function (burger) {
    burger.addEventListener("click", function () {
      const target = document.getElementById(burger.dataset.target);
      burger.classList.toggle("is-active");
      if (target) target.classList.toggle("is-active");
    });
  });

  // Copy BibTeX
  const btn = document.getElementById("copy-bibtex");
  const pre = document.getElementById("bibtex");
  if (btn && pre) {
    btn.addEventListener("click", function () {
      navigator.clipboard.writeText(pre.innerText.trim()).then(function () {
        const old = btn.innerHTML;
        btn.innerHTML = '<span class="icon"><i class="fas fa-check"></i></span><span>Copied!</span>';
        setTimeout(function () { btn.innerHTML = old; }, 1600);
      });
    });
  }
});
