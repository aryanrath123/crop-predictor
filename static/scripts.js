document
  .getElementById("cropForm")
  .addEventListener("submit", async (event) => {
    event.preventDefault(); // Prevent default form submission

    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());
    const resultDiv = document.getElementById("result");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (response.ok) {
        resultDiv.textContent = `Recommended Crop: ${result.recommended_crop}`;
        resultDiv.style.color = "green";
      } else {
        resultDiv.textContent = `Error: ${result.error}`;
        resultDiv.style.color = "red";
      }

      resultDiv.classList.add("visible");
    } catch (error) {
      resultDiv.textContent = `Error: ${error.message}`;
      resultDiv.style.color = "red";
      resultDiv.classList.add("visible");
    }
  });
