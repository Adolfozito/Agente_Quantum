import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
import google.generativeai as genai
import os
import json

# --- Configuração de Estilo e Avisos ---
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# --- INTERFACE DO STREAMLIT (Início) ---
st.set_page_config(layout="wide")
st.title("🤖 Agente de Análise de Notas Fiscais - Grupo Quantum -I2A2")

# --- CONFIGURAÇÃO DA API GEMINI (Lógica Aprimorada) ---
# Tenta obter a chave dos segredos (ideal para deploy no Streamlit Cloud)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("API Key carregada dos segredos!")
    genai.configure(api_key=GEMINI_API_KEY)
    api_key_configurada = True
except (FileNotFoundError, KeyError):
    GEMINI_API_KEY = None
    api_key_configurada = False

if not api_key_configurada:
    st.warning("Chave de API do Gemini não encontrada nos segredos.")
    GEMINI_API_KEY = st.text_input(
        "Para continuar, por favor, cole sua Chave de API do Gemini aqui:",
        type="password",
        key="local_api_key_input"
    )
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        st.success("API Key configurada com sucesso!")
        api_key_configurada = True
    else:
        st.info("Aguardando a chave de API para continuar...")
        st.stop()


# =====================================================================================
#  FERRAMENTAS DE ANÁLISE (TOOLS)
# =====================================================================================

class DataRepository:
    def __init__(self, df=None):
        self.df = df if df is not None else pd.DataFrame()

data_repo = DataRepository()

# As funções são definidas normalmente. O Gemini irá inferir os parâmetros
# a partir da assinatura da função e da docstring.
def calcular_valor_total_gasto() -> str:
    """
    Calcula e retorna o valor total de todas as notas fiscais no DataFrame.
    Use esta função para perguntas sobre o montante total, gasto total, etc.
    """
    if data_repo.df.empty:
        return json.dumps({"erro": "DataFrame vazio ou não carregado."})
    
    total = data_repo.df['VALOR TOTAL'].sum()
    return json.dumps({"valor_total_gasto": f"{total:,.2f}"})

def encontrar_item_mais_caro_ou_barato(ordenacao: str = 'caro') -> str:
    """
    Encontra o item com o maior ou menor valor unitário.
    :param ordenacao: str, 'caro' para o maior valor, 'barato' para o menor valor.
    """
    if data_repo.df.empty:
        return json.dumps({"erro": "DataFrame vazio."})

    if ordenacao == 'caro':
        idx = data_repo.df['VALOR UNITÁRIO'].idxmax()
        desc = "mais caro"
    else:
        idx = data_repo.df['VALOR UNITÁRIO'].idxmin()
        desc = "mais barato"
        
    item = data_repo.df.loc[idx]
    resultado = {
        "item": item['ITEM'],
        "fornecedor": item['FORNECEDOR'],
        "valor_unitario": f"{item['VALOR UNITÁRIO']:,.2f}",
        "descricao": desc
    }
    return json.dumps(resultado)

def obter_top_n_itens_por(metrica: str, n: int = 5, ordem: str = 'maior') -> str:
    """
    Obtém os 'n' itens principais com base em uma métrica (quantidade ou valor).
    :param metrica: str, a métrica para agrupar. Deve ser 'quantidade' ou 'valor'.
    :param n: int, o número de itens a retornar. Padrão é 5.
    :param ordem: str, 'maior' para os maiores valores (top), 'menor' para os menores (bottom).
    """
    try:
        n = int(n)
    except (ValueError, TypeError):
        return json.dumps({"erro": f"O parâmetro 'n' para a contagem deve ser um número inteiro. Recebido: {n}"})

    if data_repo.df.empty:
        return json.dumps({"erro": "DataFrame vazio."})

    if metrica == 'quantidade':
        agrupado = data_repo.df.groupby('ITEM')['QUANTIDADE'].sum()
    elif metrica == 'valor':
        agrupado = data_repo.df.groupby('ITEM')['VALOR TOTAL'].sum()
    else:
        return json.dumps({"erro": "Métrica inválida. Use 'quantidade' ou 'valor'."})

    if ordem == 'maior':
        resultado = agrupado.nlargest(n)
    else:
        resultado = agrupado.nsmallest(n)

    return resultado.to_json()


def obter_top_n_fornecedores_por_valor(n: int = 5, ordem: str = 'maior') -> str:
    """
    Obtém os 'n' fornecedores principais com base no valor total recebido.
    :param n: int, o número de fornecedores a retornar. Padrão é 5.
    :param ordem: str, 'maior' para os que mais receberam, 'menor' para os que menos receberam.
    """
    try:
        n = int(n)
    except (ValueError, TypeError):
        return json.dumps({"erro": f"O parâmetro 'n' para a contagem deve ser um número inteiro. Recebido: {n}"})

    if data_repo.df.empty:
        return json.dumps({"erro": "DataFrame vazio."})

    agrupado = data_repo.df.groupby('FORNECEDOR')['VALOR NOTA FISCAL'].sum()
    
    if ordem == 'maior':
        resultado = agrupado.nlargest(n)
    else:
        resultado = agrupado.nsmallest(n)

    return resultado.to_json()


def contar_notas_ou_itens(contar_o_que: str, fornecedor: str = None) -> str:
    """
    Conta o número total de notas fiscais únicas ou itens.
    Pode filtrar por um fornecedor específico se o nome for fornecido.
    :param contar_o_que: str, o que contar. Deve ser 'notas' ou 'itens'.
    :param fornecedor: str, opcional, nome do fornecedor para filtrar a contagem.
    """
    if data_repo.df.empty:
        return json.dumps({"erro": "DataFrame vazio."})
    
    df_filtrado = data_repo.df
    if fornecedor:
        df_filtrado = data_repo.df[data_repo.df['FORNECEDOR'].str.contains(fornecedor, case=False, na=False)]
        if df_filtrado.empty:
            return json.dumps({"erro": f"Nenhum fornecedor encontrado com o nome '{fornecedor}'."})
    
    if contar_o_que == 'notas':
        contagem = df_filtrado['CHAVE DE ACESSO'].nunique()
    elif contar_o_que == 'itens':
        contagem = len(df_filtrado)
    else:
        return json.dumps({"erro": "Parâmetro 'contar_o_que' inválido. Use 'notas' ou 'itens'."})
        
    return json.dumps({"contagem": contagem})


lista_de_ferramentas = [
    calcular_valor_total_gasto,
    encontrar_item_mais_caro_ou_barato,
    obter_top_n_itens_por,
    obter_top_n_fornecedores_por_valor,
    contar_notas_ou_itens,
]


# --- CLASSE DO AGENTE ---
class AgenteNotasFiscais:
    def __init__(self, df_cabecalho_input, df_itens_input):
        self.df_consolidado = pd.DataFrame()
        self._load_and_preprocess_data(df_cabecalho_input, df_itens_input)
        data_repo.df = self.df_consolidado
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            tools=lista_de_ferramentas
        )
        self.chat = self.model.start_chat(enable_automatic_function_calling=True)

    def _load_and_preprocess_data(self, df_cabecalho, df_itens):
        st.write("Carregando e pré-processando dados...")
        if df_cabecalho is None or df_itens is None:
            st.error("Dados de cabeçalho ou itens não fornecidos.")
            return

        df_cabecalho.columns = [str(col).strip().upper() for col in df_cabecalho.columns]
        df_itens.columns = [str(col).strip().upper() for col in df_itens.columns]

        df_cabecalho.rename(columns={'RAZÃO SOCIAL EMITENTE': 'FORNECEDOR'}, inplace=True)
        df_itens.rename(columns={'DESCRIÇÃO DO PRODUTO/SERVIÇO': 'ITEM'}, inplace=True)

        for col in ['VALOR NOTA FISCAL']:
            if col in df_cabecalho.columns:
                df_cabecalho[col] = pd.to_numeric(
                    df_cabecalho[col].astype(str).str.replace(',', '.', regex=False), errors='coerce'
                )

        for col in ['QUANTIDADE', 'VALOR UNITÁRIO', 'VALOR TOTAL']:
            if col in df_itens.columns:
                df_itens[col] = pd.to_numeric(
                    df_itens[col].astype(str).str.replace(',', '.', regex=False), errors='coerce'
                )
        
        if 'CHAVE DE ACESSO' not in df_cabecalho.columns or 'CHAVE DE ACESSO' not in df_itens.columns:
            st.error("A coluna 'CHAVE DE ACESSO' é essencial para a junção e não foi encontrada em um dos arquivos.")
            return
            
        df_cabecalho['CHAVE DE ACESSO'] = df_cabecalho['CHAVE DE ACESSO'].astype(str)
        df_itens['CHAVE DE ACESSO'] = df_itens['CHAVE DE ACESSO'].astype(str)

        cols_nos_itens_que_existem_no_cabecalho = set(df_itens.columns) & set(df_cabecalho.columns)
        cols_para_remover = [col for col in cols_nos_itens_que_existem_no_cabecalho if col != 'CHAVE DE ACESSO']
        df_itens_clean = df_itens.drop(columns=cols_para_remover, errors='ignore')

        self.df_consolidado = pd.merge(df_cabecalho, df_itens_clean, on='CHAVE DE ACESSO', how='inner')
        self.df_consolidado.fillna({'ITEM': 'Não especificado', 'FORNECEDOR': 'Não especificado'}, inplace=True)
        self.df_consolidado.fillna(0, inplace=True)

        st.success("Dados carregados e pré-processados com sucesso!")
        st.write(f"Total de {len(self.df_consolidado)} registros de itens consolidados.")

    def consultar_ia(self, query):
        if self.df_consolidado.empty:
            return "Os dados ainda não foram carregados ou processados."

        try:
            with st.spinner("O agente de IA está pensando..."):
                response = self.chat.send_message(query)
                return response.text
        except Exception as e:
            st.error(f"Ocorreu um erro ao consultar a IA: {e}")
            return "Desculpe, não consegui processar sua pergunta."

# --- SEÇÃO PRINCIPAL DA INTERFACE ---
st.header("1. Upload dos Arquivos de Notas Fiscais")
col1, col2 = st.columns(2)
with col1:
    uploaded_cabecalho_file = st.file_uploader("Arquivo de Cabeçalho (CSV)", type="csv")
with col2:
    uploaded_itens_file = st.file_uploader("Arquivo de Itens (CSV)", type="csv")

def load_dataframe(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file, sep=';', encoding='latin1')
    except Exception:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
        except Exception as e:
            st.error(f"Não foi possível ler o arquivo {uploaded_file.name}. Verifique o formato. Erro: {e}")
            return None

if 'agente' not in st.session_state:
    st.session_state.agente = None

if uploaded_cabecalho_file and uploaded_itens_file:
    if st.button("Analisar Arquivos e Iniciar Agente"):
        df_cabecalho = load_dataframe(uploaded_cabecalho_file)
        df_itens = load_dataframe(uploaded_itens_file)

        if df_cabecalho is not None and df_itens is not None:
            with st.spinner("Inicializando o agente com os dados..."):
                st.session_state.agente = AgenteNotasFiscais(df_cabecalho, df_itens)
        else:
            st.error("Falha ao carregar um ou ambos os arquivos. O agente não pode ser iniciado.")

st.header("2. Converse com o Agente")
st.sidebar.header("Exemplos de Perguntas")
st.sidebar.info("""
- Qual o gasto total?
- Qual o item mais caro e qual o seu fornecedor?
- Quais os 5 fornecedores que mais receberam dinheiro?
- Liste os 10 itens com maior quantidade vendida.
- Quantas notas fiscais foram emitidas pela 'NOME DA EMPRESA LTDA'? (Substitua pelo nome real)
- Quantos itens no total foram comprados?
""")

if st.session_state.agente:
    st.success("Agente pronto! Faça sua pergunta abaixo.")
    user_query = st.text_input("Pergunte sobre as notas fiscais:", key="query_input")
    if user_query:
        response_text = st.session_state.agente.consultar_ia(user_query)
        st.markdown("---")
        st.subheader("Resposta do Agente:")
        st.markdown(response_text)
else:
    st.info("Faça o upload dos arquivos e clique em 'Analisar' para começar.")
