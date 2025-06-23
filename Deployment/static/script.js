document.addEventListener('DOMContentLoaded', function() {
    // Set default values for date input
    const dateInput = document.getElementById('date');
    const today = new Date();
    dateInput.value = today.toISOString().split('T')[0];
    
    // Set default values for competition and promo2 fields based on date
    const currentYear = today.getFullYear();
    document.getElementById('competitionOpenSinceYear').value = currentYear - 2;
    document.getElementById('promo2SinceYear').value = currentYear;
    document.getElementById('promo2SinceWeek').value = Math.ceil((today - new Date(currentYear, 0, 1)) / (7 * 24 * 60 * 60 * 1000));

    // Add event listener for Promo2 selection
    const promo2Select = document.getElementById('promo2');
    const promo2Fields = document.querySelectorAll('.promo2-fields');

    function togglePromo2Fields() {
        const isPromo2 = promo2Select.value === '1';
        promo2Fields.forEach(field => {
            field.style.display = isPromo2 ? 'flex' : 'none';
        });

        // Reset Promo2 related fields if Promo2 is set to No
        if (!isPromo2) {
            document.getElementById('promo2SinceWeek').value = '';
            document.getElementById('promo2SinceYear').value = '';
            document.getElementById('promoInterval').value = '';
        }
    }

    // Initial toggle on page load
    togglePromo2Fields();

    // Add event listener for changes
    promo2Select.addEventListener('change', togglePromo2Fields);
});

document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get the date and calculate day of week (1 = Monday, 7 = Sunday)
    const date = document.getElementById('date').value;
    const jsDate = new Date(date);
    const dayOfWeek = jsDate.getDay() || 7; // Convert Sunday from 0 to 7
    
    const formData = {
        store: parseInt(document.getElementById('store').value),
        date: date,
        dayofweek: dayOfWeek,
        open: parseInt(document.getElementById('open').value),
        promo: parseInt(document.getElementById('promo').value),
        stateholiday: document.getElementById('stateHoliday').value,
        schoolholiday: parseInt(document.getElementById('schoolHoliday').value),
        storetype: document.getElementById('storeType').value,
        assortment: document.getElementById('assortment').value,
        competitiondistance: parseFloat(document.getElementById('competitionDistance').value) || 0,
        competitionopensincemonth: parseInt(document.getElementById('competitionOpenSinceMonth').value) || 0,
        competitionopensinceyear: parseInt(document.getElementById('competitionOpenSinceYear').value) || 0,
        promo2: parseInt(document.getElementById('promo2').value),
        promo2sinceweek: parseInt(document.getElementById('promo2SinceWeek').value) || 0,
        promo2sinceyear: parseInt(document.getElementById('promo2SinceYear').value) || 0,
        promointerval: document.getElementById('promoInterval').value
    };

    // If competition distance is 0, set competition dates to 0
    if (formData.competitiondistance === 0) {
        formData.competitionopensincemonth = 0;
        formData.competitionopensinceyear = 0;
    }

    // If store doesn't participate in Promo2, set related fields to 0
    if (formData.promo2 === 0) {
        formData.promo2sinceweek = 0;
        formData.promo2sinceyear = 0;
        formData.promointerval = '';
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        if (result.predicted_sales !== undefined) {
            document.getElementById('result').style.display = 'block';
            document.getElementById('salesValue').textContent = 
                result.predicted_sales.toLocaleString(undefined, {
                    minimumFractionDigits: 0,
                    maximumFractionDigits: 0
                });
        } else {
            alert('Error: ' + (result.error || 'Unknown error occurred'));
        }
    } catch (error) {
        console.error('Error details:', error);
        alert('Error making prediction: ' + error.message);
    }
}); 