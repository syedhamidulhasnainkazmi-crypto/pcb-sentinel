// PCB Sentinel - Workspace upload/detect logic
document.addEventListener('DOMContentLoaded', function () {
  const dropzone = document.getElementById('dropzone');
  const imageInput = document.getElementById('imageInput');
  const detectBtn = document.getElementById('detectBtn');
  const status = document.getElementById('status');
  const resultsArea = document.getElementById('resultsArea');
  const resultImage = document.getElementById('resultImage');
  const defectsList = document.getElementById('defectsList');

  if (!dropzone) return; // not on workspace page

  let selectedFile = null;

  dropzone.addEventListener('click', () => imageInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      selectedFile = e.dataTransfer.files[0];
      onFileSelected();
    }
  });

  imageInput.addEventListener('change', () => {
    if (imageInput.files.length) {
      selectedFile = imageInput.files[0];
      onFileSelected();
    }
  });

  function onFileSelected() {
    status.textContent = `Selected: ${selectedFile.name}`;
    status.style.color = 'var(--text-primary)';
    detectBtn.disabled = false;
  }

  detectBtn.addEventListener('click', () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('image', selectedFile);

    detectBtn.disabled = true;
    status.textContent = '⏳ Processing...';
    status.style.color = 'var(--warn)';
    resultsArea.style.display = 'none';

    fetch('/detect', { method: 'POST', body: formData })
      .then((res) => res.json())
      .then((data) => {
        detectBtn.disabled = false;
        if (data.error) {
          status.textContent = '❌ ' + data.error;
          status.style.color = 'var(--danger)';
          return;
        }

        status.textContent = `✅ Done in ${data.processing_time.toFixed(0)}ms — ${data.scans_remaining !== null ? data.scans_remaining + ' scans left' : 'unlimited scans'}`;
        status.style.color = 'var(--success)';

        resultImage.src = data.result_image_url + '?t=' + Date.now();
        resultsArea.style.display = 'block';

        if (data.defect_count === 0) {
          defectsList.innerHTML = '<div class="defect-row"><span>✅ No defects detected — board is clean</span></div>';
        } else {
          defectsList.innerHTML = data.defects.map((d, i) => `
            <div class="defect-row">
              <span>${i + 1}. ${d.class}</span>
              <span class="severity-pill ${d.severity}">${d.severity.toUpperCase()} · ${(d.confidence * 100).toFixed(0)}%</span>
            </div>
          `).join('');
        }
      })
      .catch((err) => {
        detectBtn.disabled = false;
        status.textContent = '❌ Error: ' + err.message;
        status.style.color = 'var(--danger)';
      });
  });
});
