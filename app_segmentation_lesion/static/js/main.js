const form = document.getElementById('upload-form');
const imageInput = document.querySelector('input[type="file"]');
const imagePreview = document.getElementById('imagePreview');
const overlayImage = document.getElementById('overlayImage'); // Agrega esta línea para definir overlayImage
const maskImage = document.getElementById('maskImage'); // Agrega esta línea para definir maskImage
const pointsImage = document.getElementById('pointsImage'); // Agrega esta línea para definir pointsImage
// const detectedImage = document.getElementById('detectedImage');
const downloadLink = document.getElementById('downloadLink');
const spinner = document.getElementById('spinner');


// Ocultar el enlace de descarga al inicio
downloadLink.style.display = "none";

// Funcionalidad para mostrar una vista previa de la imagen seleccionada
imageInput.addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = "block";
        }
        reader.readAsDataURL(file);
    } else {
        imagePreview.style.display = "none";
    }
});

// Función para verificar si hay imágenes en la carpeta 'static/detect_results'
function checkForImages(attempts) {
    fetch('/check_results')
    .then(response => response.json())
    .then(data => {
        if (data.has_images) {
            // Imágenes encontradas, esperar 5 segundos antes de mostrarlas
            setTimeout(() => {
                spinner.style.display = 'none';
                overlayImage.src = '/static/' + data.overlay_path;

                console.log('/static/' + data.overlay_path)
                maskImage.src = '/static/' + data.mask_path;
                console.log('/static/' + data.mask_path)
                pointsImage.src = '/static/' + data.bbox_path;


                downloadLink.href = `/static/${data.overlay_path.split('/').pop()}`;
                downloadLink.style.display = "block";
                overlayImage.style.display = "block";
                maskImage.style.display = "block";
                pointsImage.style.display = "block";
            }, 5000);
        } else if (attempts >= 2) {
            // No se encontraron imágenes después de 5 intentos
            spinner.style.display = 'none';
            document.getElementById('detectionResult').textContent = "Fallo en la detección";
        } else {
            // Reintentar después de 1 segundo
            setTimeout(() => checkForImages(attempts + 1), 1000);
        }
    });
}




// Funcionalidad para enviar la imagen para detección
form.addEventListener('submit', function(e) {
    e.preventDefault();

    // Mostrar el spinner
    spinner.style.display = 'block';

    const formData = new FormData(this);

    fetch('/detect_lesion', {
        method: 'POST',
        body: formData
    })
    .then(() => {
        // Iniciar la verificación de imágenes
        checkForImages(0);
    })
    .catch(error => {
        spinner.style.display = 'none';
        console.error('Error:', error);
    });
});


// Función para verificar si hay imágenes en la carpeta 'static/detect_results'
// function checkForImages(attempts) {
//     fetch('/check_results')
//     .then(response => response.json())
//     .then(data => {
//         if (data.has_images) {
//             // Imágenes encontradas, esperar 5 segundos antes de mostrarlas
//             setTimeout(() => {
//                 spinner.style.display = 'none';
//                 detectedImage.src = '/static/' + data.image_path;
//                 detectedImage.style.display = "block";
//                 downloadLink.href = `/static/${data.image_path.split('/').pop()}`;
//                 downloadLink.style.display = "block";
//             }, 5000);
//         } else if (attempts >= 2) {
//             // No se encontraron imágenes después de 5 intentos
//             spinner.style.display = 'none';
//             document.getElementById('detectionResult').textContent = "Fallo en la detección";
//         } else {
//             // Reintentar después de 1 segundo
//             setTimeout(() => checkForImages(attempts + 1), 1000);
//         }
//     });
// }
