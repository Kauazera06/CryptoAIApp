import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from crypto_ai_advanced import build_and_run, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, MODEL_FILE_DEFAULT
import numpy as np

st.set_page_config(page_title="Crypto AI Advanced", layout="wide")
st.title("💹 Crypto AI Advanced - Streamlit Interface")
st.markdown("Interface para rodar o pipeline de análise de criptomoedas com LightGBM e indicadores técnicos — agora com **previsão de sinais, preços estimados e confiança do modelo**.")

# ------------------------------
# Inputs do usuário
# ------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    symbol = st.text_input("Symbol (ex: BTC/USDT, ETH/USDT):", value=DEFAULT_SYMBOL).upper()
with col2:
    timeframe = st.selectbox("Timeframe:", ["1m","5m","15m","30m","1h","4h","1d"], index=4)
with col3:
    limit = st.slider("Quantidade de velas:", min_value=200, max_value=5000, value=2000, step=100)
with col4:
    tune = st.checkbox("Tunar modelo (Optuna)", value=False)

no_backtest = st.checkbox("Pular backtest", value=False)
col5, col6 = st.columns(2)
with col5:
    forward_horizon = st.number_input("Forward Horizon (velas futuras):", min_value=1, max_value=100, value=12)
with col6:
    predict_future = st.checkbox("Prever próximas velas", value=True)

threshold_up = st.number_input("Threshold UP:", value=0.02, step=0.01, format="%.4f")
threshold_down = st.number_input("Threshold DOWN:", value=-0.02, step=0.01, format="%.4f")

# ------------------------------
# Funções auxiliares
# ------------------------------
def color_signal(val):
    if val == 1:
        color = 'green'
    elif val == -1:
        color = 'red'
    else:
        color = 'gray'
    return f'color: {color}; font-weight:bold'

def estimate_future_prices(df, n_future=12):
    """Estima preços futuros com base na média e volatilidade recente."""
    recent_close = df["close"].tail(20)
    last_price = recent_close.iloc[-1]
    avg_return = recent_close.pct_change().mean()
    volatility = recent_close.pct_change().std()

    prices = []
    current_price = last_price
    for _ in range(n_future):
        simulated_change = np.random.normal(avg_return, volatility)
        current_price *= (1 + simulated_change)
        prices.append(current_price)
    return prices

# ------------------------------
# Botão para rodar
# ------------------------------
if st.button("🔄 Executar análise"):
    with st.spinner("Rodando pipeline avançado... isso pode demorar alguns minutos dependendo do número de velas..."):
        result = build_and_run(symbol=symbol, timeframe=timeframe, limit=limit,
        tune=tune, forward_horizon=forward_horizon,
        threshold_up=threshold_up, threshold_down=threshold_down,
        model_file=MODEL_FILE_DEFAULT,
        no_backtest=no_backtest,
        predict_future=predict_future)
    st.success("✅ Pipeline executado!")

    df = result['df']
    future_df = result.get('future_df', None)
    model = result['model']
    label_map = result['label_map']
    feature_cols = result['feature_cols']

    # ------------------------------
    # Estimar preços futuros + confiança
    # ------------------------------
    if predict_future and future_df is not None and not future_df.empty:
        estimated_prices = estimate_future_prices(df, n_future=len(future_df))
        future_df["preco_estimado"] = estimated_prices

        # calcular confiança das previsões
        X_future = df[feature_cols].iloc[-1:].values
        preds_proba = model.predict(X_future)
        inv_label_map = {v: k for k, v in label_map.items()}
        confs = []
        for _, row in future_df.iterrows():
            # mesma confiança para exemplo — em prática você pode recalcular iterativamente
            top_conf = preds_proba.max() * 100
            confs.append(round(top_conf, 2))
        future_df["confianca_%"] = confs

    # ------------------------------
    # Gráfico de velas interativo
    # ------------------------------
    st.subheader(f"🕯️ Gráfico de Velas - {symbol} ({timeframe})")
    try:
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        )])

        # Adiciona sinais do modelo (históricos)
        if 'sinal_previsto' in df.columns:
            buy_signals = df[df['sinal_previsto'] == 1]
            sell_signals = df[df['sinal_previsto'] == -1]

            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['close'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=12),
                name='BUY'
            ))

            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['close'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=12),
                name='SELL'
            ))

        # Adiciona previsões futuras (em azul)
        if predict_future and future_df is not None and not future_df.empty:
            future_buy = future_df[future_df["sinal_previsto"] == 1]
            future_sell = future_df[future_df["sinal_previsto"] == -1]
            future_hold = future_df[future_df["sinal_previsto"] == 0]

            if not future_buy.empty:
                fig.add_trace(go.Scatter(
                    x=future_buy.index,
                    y=future_buy["preco_estimado"],
                    mode='markers+lines',
                    marker=dict(symbol='triangle-up', color='blue', size=12),
                    name='Future BUY'
                ))

            if not future_sell.empty:
                fig.add_trace(go.Scatter(
                    x=future_sell.index,
                    y=future_sell["preco_estimado"],
                    mode='markers+lines',
                    marker=dict(symbol='triangle-down', color='blue', size=12),
                    name='Future SELL'
                ))

            if not future_hold.empty:
                fig.add_trace(go.Scatter(
                    x=future_hold.index,
                    y=future_hold["preco_estimado"],
                    mode='markers+lines',
                    marker=dict(symbol='circle', color='gray', size=10),
                    name='Future HOLD'
                ))

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            yaxis_title="Preço (USDT)",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao gerar gráfico de velas: {e}")

    # ------------------------------
    # Performance do Backtest
    # ------------------------------
    if result.get("performance") is not None:
        st.subheader("📊 Performance do Backtest")
        st.write(f"{result['performance']*100:.2f}% de acerto")

        # Equity curve (simulada cumulativa)
        st.subheader("📈 Equity Curve")
        try:
            equity_curve = (df['sinal_previsto'] == df['label']).astype(int).cumsum()
            fig, ax = plt.subplots(figsize=(10,5))
            equity_curve.plot(ax=ax)
            ax.set_title(f"Equity Curve - {symbol} {timeframe}")
            ax.set_ylabel("Equity (acertos acumulados)")
            ax.set_xlabel("Tempo")
            st.pyplot(fig)
        except Exception:
            st.info("Equity Curve não disponível.")

    # ------------------------------
    # Últimas velas e sinais
    # ------------------------------
    st.subheader("📝 Últimas velas e Sinais")
    st.dataframe(df[['open','high','low','close','volume','label','sinal_previsto']].tail(20)
    .style.applymap(color_signal, subset=['sinal_previsto']))

    # ------------------------------
    # Previsões futuras (tabela)
    # ------------------------------
    if predict_future and future_df is not None and not future_df.empty:
        st.subheader("🔮 Previsões Futuras (IA)")
        st.dataframe(
            future_df[["sinal_previsto", "preco_estimado", "confianca_%"]]
            .rename(columns={
                "sinal_previsto": "Sinal Previsto",
                "preco_estimado": "Preço Estimado (USDT)",
                "confianca_%": "Confiança (%)"
            })
            .style.applymap(color_signal, subset=["Sinal Previsto"])
        )
