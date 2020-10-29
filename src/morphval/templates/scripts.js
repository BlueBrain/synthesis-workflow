(function() {
  function setupNavBar() {
    var nav_toggle = document.getElementById('nav_toggle');
    if (nav_toggle){
      nav_toggle.addEventListener('click', function () {
        this.classList.toggle('nav-close');
        if ( this.innerText === '>' ? this.innerText = 'X' : this.innerText = '>');
        document.getElementById('nav_items_container').classList.toggle('line');
        document.getElementById('nav_container').classList.toggle('line');
        document.getElementById('section_container').classList.toggle('full-screen');
        // change the container size because viewport is larger
        setContainerHeight();
      });

      document.getElementById('scroll_top').addEventListener('click', function () {
        window.scrollTo(0,0);
      });
    }
  }

  function addColorsSummary() {
    document.querySelectorAll('.status-colored').forEach(function (elem) {
      if (elem.innerText === 'FAIL') {
        elem.classList.add('status-fail');
      }
      else if (elem.innerText === 'PASS') {
        elem.classList.add('status-pass');
      }
    });
  }

  function setContainerHeight() {
    // change the container height based on the height of the images
    // set the height to avoid arrows to move when image is loaded
    var containers = document.querySelectorAll('.section-carousel-container');
    var imageSize = document.querySelector('.carousel-active').height;
    containers.forEach(function(container) {
      container.style.height = imageSize + 'px';
    })
  }

  function addCarrousel() {
    var arrowDisabledName = 'arrow-disabled';
    var leftArrowName = 'left-arrow';
    var rightArrowName = 'right-arrow';
    var carouselActiveName = 'carousel-active';
    var carouselArrowsClass = '.carousel-arrow';
    var imagesContainerClass = '.img-container';
    var carouselTemplateClass = '.carousel-template'
    var imgCarouselName = 'img-carousel';
    var carouselTitleClass = '.carousel-title';
    var referenceMorphologies = 'ref-cell-files';
    var testMorphologies = 'test-cell-files';
    var leftArrows = document.querySelectorAll('.' + leftArrowName);
    var rightArrows = document.querySelectorAll('.' + rightArrowName);
    var firstLoad = true;

    // carousel that will be used for all the sections
    var template = document.querySelector(carouselTemplateClass);

    // to load the images ondemand
    function createHiddenImageCarousel (elem) {
      var hiddenImg = document.createElement('img');
      hiddenImg.src = '';
      hiddenImg.name = elem;
      hiddenImg.classList.add(imgCarouselName);
      return hiddenImg;
    }

    function copyTemplate(template, title, imgsList) {
      var template = template.cloneNode(true);
      template.querySelector(carouselTitleClass).innerText = title;
      var testCellContainer = template.querySelector(imagesContainerClass);
      imgsList.forEach(function (elem) {
        var hiddenImg = createHiddenImageCarousel(elem);
        testCellContainer.appendChild(hiddenImg);
      });
      // load the first image
      var firstImg = testCellContainer.firstChild;
      firstImg.classList.add(carouselActiveName);
      firstImg.src = firstImg.name;
      firstImg.onload = function() {
        if(firstLoad) {
          setContainerHeight();
          firstLoad = false;
        }
      }
      template.removeAttribute('hidden');
      return template;
    }

    function createMultipleCarousel(className, title) {
      // creates all the carousel for Ref o Test depending on the className
      document.querySelectorAll('.' + className).forEach(function(elem) {
        var list = JSON.parse(elem.innerText); // list of all images for that carousel
        var newTemp = copyTemplate(template, title, list);
        elem.parentElement.appendChild(newTemp);
      })
    }

    createMultipleCarousel(referenceMorphologies, 'Reference Morphologies');
    createMultipleCarousel(testMorphologies, 'Test Morphologies');

    function createArrowCarousel () {

      function addListenersCarousel () {
        var arrows = document.querySelectorAll(carouselArrowsClass);
        arrows.forEach(function (elem) {
          elem.addEventListener('click', function () {
            // load the image on demand and update the arrows
            if (this.classList.contains(leftArrowName)) {
              var container = this.parentElement;
              var currentImg = container.querySelector('.' + carouselActiveName);
              var prevImg = currentImg.previousElementSibling;
              if (prevImg) {
                currentImg.classList.remove(carouselActiveName);
                currentImg.removeAttribute('src');
                prevImg.setAttribute('src', prevImg.name);
                prevImg.onload = function() {
                  this.classList.add(carouselActiveName);
                  checkNextPrevArrows(container);
                };
              }
            }
            if (this.classList.contains(rightArrowName)) {
              var container = this.parentElement;
              var currentImg = container.querySelector('.' + carouselActiveName);
              var nextImg = currentImg.nextElementSibling;
              if (nextImg) {
                currentImg.classList.remove(carouselActiveName);
                currentImg.removeAttribute('src');
                nextImg.setAttribute('src', nextImg.name);
                nextImg.onload = function() {
                  this.classList.add(carouselActiveName);
                  checkNextPrevArrows(container);
                }
              }
            }
          });
        });
      }

      function checkNextPrevArrows (container) {
        // disable or enable the arrows based on siblings
        var currentImage = container.querySelector('.' + carouselActiveName);
        var leftArrow = container.querySelector('.' + leftArrowName);
        var rightArrow = container.querySelector('.' + rightArrowName);
        leftArrow.classList.add(arrowDisabledName);
        rightArrow.classList.add(arrowDisabledName);

        if (currentImage && currentImage.previousElementSibling) {
          leftArrow.classList.remove(arrowDisabledName);
        }
        if (currentImage && currentImage.nextElementSibling) {
          rightArrow.classList.remove(arrowDisabledName);
        }
      }

      function checkInitialArrows () {
        var imgCntainers = document.querySelectorAll(imagesContainerClass);
        imgCntainers.forEach(function(elem) {
          checkNextPrevArrows(elem.parentElement);
        });
      }

      addListenersCarousel();
      checkInitialArrows();
    }

    createArrowCarousel();
  }

  setupNavBar();
  addColorsSummary();
  addCarrousel();
})();
