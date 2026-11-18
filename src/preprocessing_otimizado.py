import pandas as pd
import numpy as np
import os

def prepare_v2_data():
    data_dir = 'data'
    
    # 1. Carregar as bases ORIGINAIS certas
    fight = pd.read_csv(os.path.join(data_dir, 'fight.csv'))
    event = pd.read_csv(os.path.join(data_dir, 'event.csv'))
    rounds = pd.read_csv(os.path.join(data_dir, 'round.csv'))
    fighter = pd.read_csv(os.path.join(data_dir, 'fighter.csv'))
    
    # 2. Filtrar Eventos a partir de 2016
    event['date'] = pd.to_datetime(event['date'])
    event_2016 = event[event['date'].dt.year >= 2016]
    
    # 3. Juntar lutas com eventos (apenas lutas 2016+)
    df = pd.merge(fight, event_2016[['event_id']], on='event_id', how='inner')
    
    # 4. Juntar com as estatísticas do Round 1 (round_no)
    r1 = rounds[rounds['round_no'] == 1].copy()
    df = pd.merge(df, r1, on=['fight_id', 'r_id', 'b_id'], how='inner')
    
    # 5. Calcular a Idade (Ano do evento - Data de nascimento do lutador)
    # Trazendo DOB (Date of Birth) do lutador Red
    df = pd.merge(df, fighter[['fighter_id', 'dob']], left_on='r_id', right_on='fighter_id', how='left')
    df = df.rename(columns={'dob': 'r_dob'})
    df.drop(columns=['fighter_id'], inplace=True)
    
    # Trazendo DOB do lutador Blue
    df = pd.merge(df, fighter[['fighter_id', 'dob']], left_on='b_id', right_on='fighter_id', how='left')
    df = df.rename(columns={'dob': 'b_dob'})
    df.drop(columns=['fighter_id'], inplace=True)
    
    # Adicionando o ano do evento na tabela df
    df = pd.merge(df, event[['event_id', 'date']], on='event_id', how='left')
    df['event_year'] = pd.to_datetime(df['date']).dt.year
    
    # Calculando a idade no momento da luta
    df['r_age'] = df['event_year'] - pd.to_datetime(df['r_dob']).dt.year
    df['b_age'] = df['event_year'] - pd.to_datetime(df['b_dob']).dt.year

    # 6. Criar as Variáveis Diferenciais (Red - Blue)
    df['diff_sig_landed'] = df['r_sig_landed'] - df['b_sig_landed']
    df['diff_td_success'] = df['r_td_success'] - df['b_td_success'] 
    
    # Converter tempo de controle
    def time_to_sec(val):
        if pd.isna(val): return np.nan
        parts = str(val).split(':')
        return int(parts[0]) * 60 + int(parts[1])

    df['r_ctrl_sec'] = df['r_ctrl'].apply(time_to_sec)
    df['b_ctrl_sec'] = df['b_ctrl'].apply(time_to_sec)
    df['diff_ctrl_sec'] = df['r_ctrl_sec'] - df['b_ctrl_sec']
    df['diff_sub_att'] = df['r_sub_att'] - df['b_sub_att']
    
    # NOVAS VARIÁVEIS OTIMIZADAS
    df['diff_age'] = df['r_age'] - df['b_age']
    df['diff_rev'] = df['r_rev'] - df['b_rev'] # Reversões de posição no chão
    
    # Target
    df['y_red_win'] = (df['winner_id'] == df['r_id']).astype(int)
    
    # Filtrar lutas com vencedor e que passaram do 1º round
    df = df[df['result_status'] == 'win'].copy()
    df = df[df['round'] >= 2].copy()
    
    # 7. Selecionar colunas finais e remover NAs
    covs = [
        'diff_sig_landed', 'diff_td_success', 'diff_ctrl_sec', 
        'diff_sub_att', 'diff_age', 'diff_rev'
    ]
    cols_to_keep = ['fight_id', 'y_red_win'] + covs
    
    df_clean = df[cols_to_keep].dropna().copy()
    
    # 8. Salvar base otimizada
    out_dir = os.path.join(data_dir, 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ufc_r1_otimizado.csv')
    
    df_clean.to_csv(out_path, index=False)
    
    print("Pré-processamento V2 Concluído!")
    print(f"Total de lutas na base otimizada: {len(df_clean)}")
    print(f"Base salva em: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    prepare_v2_data()