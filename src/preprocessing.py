import pandas as pd
import numpy as np
import os

# Pega o caminho absoluto da pasta onde o script está (src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Sobe uma pasta e entra em data/
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

def clean_and_prepare_data():
    print("Iniciando pré-processamento...")
    
    # 1. Carregar os dados originais
    fight_path = os.path.join(DATA_DIR, 'fight.csv')
    round_path = os.path.join(DATA_DIR, 'round.csv')
    
    fight = pd.read_csv(fight_path)
    rnd = pd.read_csv(round_path)
    
    # ... (mantenha a parte do meio com a lógica do pré-processamento 2 a 8 igual) ...
    # 2. Filtrar apenas as estatísticas do Round 1
    r1 = rnd[rnd['round_no'] == 1].copy()
    
    # 3. Fazer a junção das tabelas
    df = pd.merge(fight, r1, on=['fight_id', 'r_id', 'b_id'], how='inner')
    
    # 4. Filtrar lutas com vencedor e que passaram do 1º round
    df = df[df['result_status'] == 'win'].copy()
    df = df[df['round'] >= 2].copy()
    
    # 5. Converter tempo de controle ('MM:SS') para segundos inteiros
    def time_to_sec(val):
        if pd.isna(val):
            return np.nan
        parts = str(val).split(':')
        return int(parts[0]) * 60 + int(parts[1])

    df['r_ctrl_sec'] = df['r_ctrl'].apply(time_to_sec)
    df['b_ctrl_sec'] = df['b_ctrl'].apply(time_to_sec)
    
    # 6. Criar variável resposta Y (1 = Red venceu, 0 = Blue venceu)
    df['y_red_win'] = (df['winner_id'] == df['r_id']).astype(int)
    
    # 7. Criar Covariáveis Diferenciais (Red - Blue)
    df['diff_sig_landed'] = df['r_sig_landed'] - df['b_sig_landed']
    df['diff_kd'] = df['r_kd'] - df['b_kd']
    df['diff_td_success'] = df['r_td_success'] - df['b_td_success']
    df['diff_ctrl_sec'] = df['r_ctrl_sec'] - df['b_ctrl_sec']
    df['diff_sub_att'] = df['r_sub_att'] - df['b_sub_att']

    # 8. Diferença na precisão de golpes significativos
    r_acc = np.where(df['r_sig_atmp'] > 0, df['r_sig_landed'] / df['r_sig_atmp'], 0.0)
    b_acc = np.where(df['b_sig_atmp'] > 0, df['b_sig_landed'] / df['b_sig_atmp'], 0.0)
    df['diff_sig_acc'] = r_acc - b_acc
    
    # 9. Selecionar colunas finais e remover NAs
    covs = ['diff_sig_landed', 'diff_kd', 'diff_td_success', 'diff_ctrl_sec', 'diff_sub_att', 'diff_sig_acc']
    cols_to_keep = ['fight_id', 'weight_class', 'round', 'y_red_win'] + covs
    
    df_clean = df[cols_to_keep].dropna().copy()
    
    # 10. Salvar base processada
    out_dir = os.path.join(DATA_DIR, 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ufc_r1_clean.csv')
    
    df_clean.to_csv(out_path, index=False)
    
    print(f"Pré-processamento concluído!")
    print(f"Total de lutas na base final: {len(df_clean)}")
    print(f"Base salva em: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    clean_and_prepare_data()