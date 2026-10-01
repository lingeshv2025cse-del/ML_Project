function showFileName(input, targetId) {
    const fileName = input.files && input.files[0] ? input.files[0].name : 'No file selected';
    const target = document.getElementById(targetId);
    if (target) {
        target.textContent = fileName;
    }
}


document.addEventListener('DOMContentLoaded', function () {
    const table = document.getElementById('resultsTable');
    if (!table) {
        return;
    }

    const rows = Array.from(table.querySelectorAll('tbody tr'));
    const searchInput = document.getElementById('searchInput');
    const statusFilter = document.getElementById('statusFilter');
    const prevPageButton = document.getElementById('prevPage');
    const nextPageButton = document.getElementById('nextPage');
    const rowLimit = 10;
    let currentPage = 0;

    function applyFilter() {
        const searchText = searchInput ? searchInput.value.toLowerCase() : '';
        const selectedStatus = statusFilter ? statusFilter.value : '';

        const filtered = rows.filter((row) => {
            const text = row.textContent.toLowerCase();
            const status = row.getAttribute('data-status') || '';
            const matchesSearch = !searchText || text.includes(searchText);
            const matchesStatus = !selectedStatus || status === selectedStatus;
            return matchesSearch && matchesStatus;
        });

        const totalPages = Math.max(1, Math.ceil(filtered.length / rowLimit));
        currentPage = Math.min(currentPage, totalPages - 1);

        rows.forEach((row) => {
            row.style.display = 'none';
        });

        filtered.slice(currentPage * rowLimit, (currentPage + 1) * rowLimit).forEach((row) => {
            row.style.display = '';
        });

        if (prevPageButton) {
            prevPageButton.disabled = currentPage === 0;
        }
        if (nextPageButton) {
            nextPageButton.disabled = currentPage >= totalPages - 1;
        }
    }

    if (searchInput) {
        searchInput.addEventListener('input', () => {
            currentPage = 0;
            applyFilter();
        });
    }

    if (statusFilter) {
        statusFilter.addEventListener('change', () => {
            currentPage = 0;
            applyFilter();
        });
    }

    if (prevPageButton) {
        prevPageButton.addEventListener('click', () => {
            if (currentPage > 0) {
                currentPage -= 1;
                applyFilter();
            }
        });
    }

    if (nextPageButton) {
        nextPageButton.addEventListener('click', () => {
            const totalPages = Math.max(1, Math.ceil(rows.filter((row) => row.style.display !== 'none').length / rowLimit));
            if (currentPage < totalPages - 1) {
                currentPage += 1;
                applyFilter();
            }
        });
    }

    applyFilter();
});
