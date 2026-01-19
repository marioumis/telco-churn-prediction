// On récupère le formulaire et la zone de résultat
const form = document.getElementById("churn-form");
const resultDiv = document.getElementById("result");

// On ajoute un écouteur sur la soumission du formulaire
form.addEventListener("submit", async (event) => {
  event.preventDefault(); // empêche le rechargement de la page

  // Récupérer les valeurs de tous les champs du formulaire
  const formData = new FormData(form);
  const data = Object.fromEntries(formData.entries());

  // Convertir les champs numériques (sinon tout est string)
  data.SeniorCitizen = Number(data.SeniorCitizen);
  data.tenure = Number(data.tenure);
  data.MonthlyCharges = Number(data.MonthlyCharges);
  data.TotalCharges = Number(data.TotalCharges);

  try {
    // Appel à l'API Flask
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const txt = await response.text();
      resultDiv.textContent = "Erreur API : " + txt;
      return;
    }

    const json = await response.json();

    const probaPct = (json.churn_proba * 100).toFixed(1);

    let message;
    if (json.churn_pred === 1) {
      message = `⚠ Ce client est à risque de churn (probabilité : ${probaPct} %).`;
    } else {
      message = `✅ Ce client a peu de risque de churn (probabilité : ${probaPct} %).`;
    }

    resultDiv.textContent = message;
  } catch (error) {
    console.error(error);
    resultDiv.textContent = "Erreur lors de l'appel à l'API.";
  }
});
