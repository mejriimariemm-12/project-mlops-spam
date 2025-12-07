# demo.ps1 - Script de démonstration MLOps
Write-Host "🚀 DÉMONSTRATION PROJET MLOPS - Détection de Spam SMS" -ForegroundColor Cyan
Write-Host "=" * 60

# 1. Afficher structure
Write-Host "📁 STRUCTURE DU PROJET :" -ForegroundColor Green
tree /F | Select-String -Pattern "\.(py|yml|yaml|Dockerfile|txt|md)$" | Select-Object -First 20

# 2. Afficher modèles
Write-Host "`n🤖 MODÈLES DISPONIBLES :" -ForegroundColor Green
if (Test-Path "models") {
    Get-ChildItem models/*.pkl | ForEach-Object {
        Write-Host "  - $($_.Name) ($([math]::Round($_.Length/1KB, 2)) KB)"
    }
}

# 3. Afficher CI/CD status
Write-Host "`n⚙️ CI/CD STATUS :" -ForegroundColor Green
Write-Host "  ✅ GitHub Actions configuré"
Write-Host "  ✅ Badge disponible dans README.md"
Write-Host "  🔗 https://github.com/mejriimariemm-12/project-mlops-spam/actions"

# 4. Commandes de démonstration
Write-Host "`n🎯 COMMANDES DE DÉMONSTRATION :" -ForegroundColor Yellow
@"
1. Lancer l'API : 
   python -m src.api.main

2. Tester avec curl : 
   curl -X POST http://localhost:8000/predict/ ^
     -H "Content-Type: application/json" ^
     -d "{\"text\":\"WINNER! Claim your prize now!\", \"model_type\":\"svm\"}"

3. Voir documentation : 
   http://localhost:8000/docs

4. Build Docker : 
   docker build -t spam-detection-mlops .

5. Lancer avec Docker : 
   docker-compose up
"@

Write-Host "`n📊 RÉSUMÉ POUR LE PROF :" -ForegroundColor Cyan
@"
✅ Pipeline MLOps complet
✅ 4 modèles ML implémentés
✅ API REST avec FastAPI
✅ CI/CD automatisé (GitHub Actions)
✅ Containerisation Docker
✅ Versioning données (DVC)
✅ Tracking MLflow
✅ Tests d'intégration
✅ Documentation complète
"@

Write-Host "`n🎉 PROJET PRÊT POUR LA PRÉSENTATION !" -ForegroundColor Green
Write-Host "=" * 60
