const form = document.getElementById("predictionForm");

form.addEventListener("submit", async function (e) {

    e.preventDefault();

    const loader = document.getElementById("loader");
    const resultCard = document.getElementById("resultCard");

    loader.style.display = "block";
    resultCard.style.display = "none";

    const formData = new FormData(form);

    const data = {};

    formData.forEach((value, key) => {
        data[key] = value;
    });

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        console.log("Server Response:", result);

        loader.style.display = "none";

        // Check if Flask returned an error
        if (result.error) {

            resultCard.style.display = "block";

            document.getElementById("prediction").innerHTML =
                "Error";

            document.getElementById("confidence").innerHTML =
                result.error;

            return;
        }

        resultCard.style.display = "block";

        document.getElementById("prediction").innerHTML =
            result.prediction || "No Prediction";

        document.getElementById("confidence").innerHTML =
            `Confidence : ${result.confidence || 0}%`;

    }
    catch (error) {

        console.error(error);

        loader.style.display = "none";

        resultCard.style.display = "block";

        document.getElementById("prediction").innerHTML =
            "Error";

        document.getElementById("confidence").innerHTML =
            "Unable to connect to server";
    }

});