var globalModelBrowserPopup = null;
var globalModelBrowserPopupInner = null;
var globalModelBrowserListeners = [];
function closeModelBrowserPopup() {
  if (!globalModelBrowserPopup) return;

  globalModelBrowserPopup.style.display = "none";
}

function modelBrowserPopup(tab, contents) {
  if (!globalModelBrowserPopup) {
    globalModelBrowserPopup = document.createElement('div');
    globalModelBrowserPopup.onclick = closeModelBrowserPopup;
    globalModelBrowserPopup.classList.add('global-model-browser-popup');

    var close = document.createElement('div');
    close.classList.add('global-model-browser-popup-close');
    close.onclick = closeModelBrowserPopup;
    close.title = "Close";
    globalModelBrowserPopup.appendChild(close);

    globalModelBrowserPopupInner = document.createElement('div');
    globalModelBrowserPopupInner.onclick = function (event) {
      event.stopPropagation();
      return false;
    };
    globalModelBrowserPopupInner.classList.add('global-model-browser-popup-inner');
    globalModelBrowserPopup.appendChild(globalModelBrowserPopupInner);

    gradioApp().querySelector('.main').appendChild(globalModelBrowserPopup);
  }

  doThingsAfterPopup(tab)
  globalModelBrowserPopupInner.innerHTML = '';
  globalModelBrowserPopupInner.appendChild(contents);

  globalModelBrowserPopup.style.display = "flex";
}



function toggleSelected(button) {
  const filterButtons = document.querySelectorAll('.filter-btn');

  filterButtons.forEach(btn => {
    btn.classList.remove('selected');
  });
  button.classList.add('selected');
}

function filterImages(tab, kind, selectedTag) {
  const imageItems = document.querySelectorAll(`.image-item[data-kind="${kind}"]`);

  imageItems.forEach(item => {
    const itemTags = item.getAttribute('data-tags').split(' ');
    const shouldDisplay = selectedTag === 'all' || itemTags.includes(selectedTag);
    item.style.display = shouldDisplay ? 'block' : 'none';
  });
}

function doThingsAfterPopup(tab) {
  addFilterButtons(tab, 'checkpoint');
  addFilterButtons(tab, 'lora');
  addFilterButtons(tab, 'embedding');

  applyTextSearch(tab, 'checkpoint');
  applyTextSearch(tab, 'lora');
  applyTextSearch(tab, 'embedding');

  // addNsfwToggle()
  addImageClickListener(tab);


  applyNsfwClass(tab);
}

function addImageClickListener(tab) {
  const imageItems = document.querySelectorAll('.image-item');

  imageItems.forEach(item => {
    const selectButton = item.querySelector('#select-button');
    // const favoriteButton = item.querySelector('#favorite-btn');
    const titleElement = item.querySelector('.title').getAttribute('data-alias');
    const browserTabName = item.parentElement.parentElement.parentElement.querySelector('.heading-text').textContent
    selectButton.addEventListener('click', (event) => {
      if (browserTabName == 'CHECKPOINT Browser') {
        desiredCloudInferenceCheckpointName = titleElement;
        gradioApp().getElementById(`${tab}_change_cloud_checkpoint`).click()
      } else if (browserTabName == 'LORA Browser') {
        desiredCloudInferenceLoraName = titleElement;
        gradioApp().getElementById(`${tab}_change_cloud_lora`).click()
      } else if (browserTabName == 'EMBEDDING Browser') {
        desiredCloudInferenceEmbeddingName = titleElement;
        gradioApp().getElementById(`${tab}_change_cloud_embedding`).click()
      }
    });

    // favoriteButton.addEventListener('click', () => {
    //   desciredCloudInferenceFavoriteModelName = titleElement;
    //   gradioApp().getElementById(`${tab}_favorite`).click()
    // })
  })
}

function addFilterButtons(tab, kind) {
  const filterButtons = document.querySelectorAll(`.filter-btn[data-kind="${kind}"][data-tab="${tab}"]`);
  // Filter images based on selected filter
  filterButtons.forEach(button => {
    button.addEventListener('click', () => {
      const selectedTag = button.getAttribute('data-tag');
      filterImages(tab, kind, selectedTag);

      toggleSelected(button); // Add selected style to clicked button
    });
  });
}


function applyNsfwClass(tab) {
  const imageItems = document.querySelectorAll('.image-item');

  imageItems.forEach(item => {
    const itemTags = item.getAttribute('data-tags').split(' ');
    if (itemTags.includes('nsfw')) {
      item.classList.add('nsfw');
    } else {
      item.classList.remove('nsfw');
    }
  });
}

function applyTextSearch(tab, kind) {
  document.getElementById(`${tab}-${kind}-filter-search-input`).addEventListener(`input`, () => {
    const searchText = document.getElementById(`${tab}-${kind}-filter-search-input`).value.toLowerCase();
    const selectedTag = getSelectedTag(kind);

    const imageItems = document.querySelectorAll('.image-item');
    document.querySelectorAll(`.image-item[data-kind="${kind}"]`)
    imageItems.forEach(item => {
      const itemTags = item.getAttribute('data-tags').split(' ');
      const searchTerms = item.getAttribute('data-search-terms');
      const matchesSearch = searchTerms.toLowerCase().includes(searchText.toLowerCase());
      const matchesTag = selectedTag === 'all' || itemTags.includes(selectedTag);
      console.log(item, matchesSearch, matchesTag)
      item.style.display = matchesSearch && matchesTag ? 'block' : 'none';
    });
  });
}

// function doThingsAfterClosePopup() {
//   for (kind of ['checkpoint', 'lora', 'embedding']) {
//     searchInput.removeEventListener(`${kind}-search-input-listener`)
//   }
// }

function getSelectedTag(kind) {
  const selectedButton = document.querySelector(`.filter-btn.selected[data-kind="${kind}"]`);
  return selectedButton ? selectedButton.getAttribute('data-tag') : 'all';
}

function openInNewTab(url) {
  var win = window.open(url, '_blank');
  win.focus();
}