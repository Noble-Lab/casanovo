(function () {
    const pageMarker = "pfqnhktptyfdeomqabhq";

    window.onload = (event) => {
        let pTags = document.querySelectorAll("p");
        let findMarker = Array.from(pTags).find(
            paragraph => paragraph.textContent.includes(pageMarker)
        );

        // If the marker isn't on the page then this isn't
        // the page we're looking for
        if (findMarker == null) {
            return;
        }

        // Remove first casanovo header and page marker
        let firstHeader = document.querySelector("#casanovo > h2:first-of-type");
        firstHeader.remove();
        findMarker.remove();

        // Change second header text content
        let secondHeader = document.querySelector("#casanovo > h3:first-of-type");
        secondHeader.textContent = "Casanovo CLI Commands";
    }
})();