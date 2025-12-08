from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import pandas as pd
import json
import os

def load_and_analyze_data():
    """Charge et analyse des données simulées de spam"""
    print("=== Début du traitement des données spam ===")
    
    # Données simulées
    messages = [
        {'text': 'Gagnez 1000€ rapidement!', 'is_spam': True},
        {'text': 'Réunion équipe à 14h', 'is_spam': False},
        {'text': 'Vous avez gagné un iPhone!', 'is_spam': True},
        {'text': 'Rapport mensuel disponible', 'is_spam': False},
        {'text': 'Crédit gratuit immédiat', 'is_spam': True},
        {'text': 'Invitation déjeuner client', 'is_spam': False}
    ]
    
    # Conversion en DataFrame
    df = pd.DataFrame(messages)
    
    # Analyse simple
    total = len(df)
    spam_count = df['is_spam'].sum()
    ham_count = total - spam_count
    
    print(f"Total messages: {total}")
    print(f"Spam: {spam_count} ({spam_count/total*100:.1f}%)")
    print(f"Ham: {ham_count} ({ham_count/total*100:.1f}%)")
    
    # Sauvegarder les résultats
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_messages': int(total),
        'spam_count': int(spam_count),
        'ham_count': int(ham_count),
        'spam_percentage': float(spam_count/total*100)
    }
    
    os.makedirs('/opt/airflow/results', exist_ok=True)
    with open('/opt/airflow/results/spam_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Résultats sauvegardés dans /opt/airflow/results/spam_analysis.json")
    print("=== Fin du traitement ===")
    
    return results

def generate_report(**kwargs):
    """Génère un rapport à partir des résultats"""
    ti = kwargs['ti']
    results = ti.xcom_pull(task_ids='analyze_data')
    
    print("\n" + "="*50)
    print("RAPPORT DE CLASSIFICATION SPAM")
    print("="*50)
    print(f"Date: {results['timestamp']}")
    print(f"Messages analysés: {results['total_messages']}")
    print(f"Messages spam: {results['spam_count']}")
    print(f"Messages légitimes: {results['ham_count']}")
    print(f"Pourcentage spam: {results['spam_percentage']:.1f}%")
    print("="*50)
    
    # Classification simple
    if results['spam_percentage'] > 50:
        print("⚠️  ALERTE: Taux de spam élevé!")
    else:
        print("✅  Situation normale")
    
    return "Rapport généré avec succès"

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='spam_monitoring',
    default_args=default_args,
    description='Surveillance et analyse des messages spam',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['monitoring', 'spam', 'mlops'],
) as dag:

    start = DummyOperator(task_id='start')
    
    analyze_task = PythonOperator(
        task_id='analyze_data',
        python_callable=load_and_analyze_data,
    )
    
    report_task = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
    )
    
    end = DummyOperator(task_id='end')
    
    start >> analyze_task >> report_task >> end
