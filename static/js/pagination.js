var currentPage = 1;
var rowsPerPage = 10;
var totalItems = { total_data }; // Ganti dengan jumlah total data

function displayPagination(totalItems) {
    var totalPages = Math.ceil(totalItems / rowsPerPage);
    var paginationHTML = '';

    if (totalPages > 1) {
        for (var i = 1; i <= totalPages; i++) {
            paginationHTML += '<li class="page-item"><a class="page-link" href="#" onclick="changePage(' + i + ')">' + i + '</a></li>';
        }
        document.querySelector('#pagination ul').innerHTML = paginationHTML;
    }
}

function changePage(page) {
    var tableRows = document.querySelectorAll('#dataTable tbody tr');
    var startIndex = (page - 1) * rowsPerPage;
    var endIndex = startIndex + rowsPerPage;

    for (var i = 0; i < tableRows.length; i++) {
        if (i >= startIndex && i < endIndex) {
            tableRows[i].style.display = '';
        } else {
            tableRows[i].style.display = 'none';
        }
    }

    currentPage = page;
    updatePagination();
}

function updatePagination() {
    var paginationItems = document.querySelectorAll('#pagination .page-item');
    paginationItems[0].classList.toggle('disabled', currentPage === 1);
    paginationItems[paginationItems.length - 1].classList.toggle('disabled', currentPage === Math.ceil(totalItems / rowsPerPage));
}

document.getElementById('previous').onclick = function (event) {
    event.preventDefault();
    if (currentPage > 1) {
        changePage(currentPage - 1);
    }
};

document.getElementById('next').onclick = function (event) {
    event.preventDefault();
    if (currentPage < Math.ceil(totalItems / rowsPerPage)) {
        changePage(currentPage + 1);
    }
};

// Memanggil fungsi untuk menampilkan pagination saat halaman dimuat
document.addEventListener('DOMContentLoaded', function() {
    displayPagination(totalItems);
    changePage(1); // Menampilkan halaman pertama secara default
});