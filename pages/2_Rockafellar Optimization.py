import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import time 
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Importing tickers from a csv
# tickers_1 = pd.read_csv("/Users/pafornwl/Desktop/Code/Python/Streamlit/Projet2/tickers.csv", sep= ';') # Guillaume
tickers_1 = pd.read_csv("C:/Users/berlu/OneDrive/Bureau/Projet PM/1/tickers.csv", sep=";") # Louis
tickers_1 = tickers_1.set_index('Symbol')

# Importing data from Yahoo Finance
def importing_price(tickers, time_horizon,years: int = 5):
    all_data = []

    for ticker in tickers:
        raw = yf.download(
            ticker,
            period=f"{years}y",
            interval=time_horizon,
            auto_adjust=True,
            progress=False
        )

        if not raw.empty and "Close" in raw.columns:
            # We rename the series of closing prices
            close_series = raw["Close"]
            close_series.name = ticker
            all_data.append(close_series)
        else:
            st.warning(f"No data for {ticker}")

        time.sleep(0.5)  # Small pause for the rate-limit

    # Concatenate all data in one dataframe
    if all_data:
        data = pd.concat(all_data, axis=1)

        # Same temporary base
        end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        start_date = end_date - pd.DateOffset(years=years)
        date_index = pd.date_range(start=start_date, end=end_date, freq='B')

        # Reindexation
        data = data.reindex(date_index)
        data = data.dropna(how="all")
        return data
    else:
        return pd.DataFrame()
        

# User interform 
st.title('📈 Rockafellar Optimization')

st.caption("See how you can optimize your Portfolio using CVaR")

# Principal Parameters for VaR and CVaR
st.write('### Principal Parameters')
col1, col2 = st.columns(2)

# Mapping horizon labels to trading days
horizon_to_days = {
    '1 Day': '1d',
    '1 Week': '5d'
}

# Choosing the horizon
h_array = ['1 Day', '1 Week']
horizon = col1.selectbox('Select your Time Horizon', h_array)
time_horizon = horizon_to_days.get(horizon)

alpha = float(col2.number_input('Enter your Confidence Level (in %) | MAX 99.97%', value = 99.0, step=1.0, max_value=99.97))

# Choosing your assets
st.write('### Assets & Initial Portfolio Value')
col1, col2 = st.columns(2)

n = int(col1.text_input('Number of assets', value = 1))
initial_investment = int(col2.number_input('Initial Investment', value=10000, step= 5000))

tick_dict = dict()
w_dict = dict()

cols = st.columns(5)

for i in range(n): 
    j = i%5
    with cols[j]:
        tick_dict[f'tick_{i}'] = st.selectbox(f"Ticker {i + 1}", tickers_1.index)
        w_dict[f'w{i}'] = st.number_input(f"Ticker {i + 1} (%)", min_value=0.0, max_value=100.0, value=100/n, step=0.1, format="%.1f")
        st.markdown(
            f"""
            <div style='margin-bottom: 10px; padding: 10px; background-color: #000002; border-radius: 1px;width: 125px; height: 70px;'>
                <a href='https://fr.finance.yahoo.com/quote/{tick_dict[f'tick_{i}']}/' target='_blank' style='text-decoration: none; display: flex; align-items: center; gap: 8px;'>
                    <span style='color: #808080; font-size: 12px; font-weight: bold, underline;'>{tickers_1.loc[tick_dict[f'tick_{i}'], "Name"]}</span>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
        
weight = list(w_dict.values())
tickers = list(tick_dict.values())

    
submitted = st.button("Confirm the weighting of your Portfolio")
 
if submitted:
    total = sum(weight)
    
    if total != 100.0:
        st.warning("Sum is not 100%. Please normalize!")

    # Adding the tickers (drop spaces + add uppers)
    tickers = [t.strip().upper() for t in tickers] 
    
    # Downloading the data
    data = importing_price(tickers, time_horizon, 5)
    data = data.loc[:, tickers] # Ordering the columns like the UI
    data_clean = data.dropna(how="any") # All the assets need to have a price in order to keep the data
    
    # Calculating the returns without taking NA
    returns = data_clean.pct_change().dropna(how="any") # No fill na, we drop missing data
    
    w = np.asarray(weight, dtype=float) / 100.0


    daily_return = (1 + returns).cumprod().shift(1).fillna(1.0).values @ w
    portfolio_value = pd.Series(initial_investment * daily_return, index=returns.index)
    portfolio_returns = portfolio_value.pct_change().dropna()
    
    # Storing the data 
    st.session_state["portfolio_returns"] = portfolio_returns
    st.session_state["returns"] = returns
    st.session_state["weight"] = weight

# Setting a new page
st.divider()
st.header("Rockafellar Optimization")

submitted3 = st.checkbox("Launch the Optimization")
    
if submitted3:
    # Getting the data stored
    returns = st.session_state["returns"] # Our J*n (days * nb of assets)
    weight = st.session_state["weight"]
    
     
    # We convert our dataframe into an array because the cvxpy and numpy libraries work more efficiently with arrays
    R = returns.values
    J, n = R.shape # We simply retrieve the number of simulation days and the number of assets in the portfolio
    
    # Page in two colums
    c1, c2 = st.columns(2)
    
    # We reuse the parameters, keeping the previously specified confidence level
    cap_min = c1.number_input(f"Lower Limit per Asset (in %) | MAX {100/n:.2f}%", min_value=0.0, max_value=round(100.0 / n, 2), value=0.0, step=0.01)
    cap_max = c2.number_input(f"Upper limit per Asset (in %) | MIN {100/n:.2f}%", min_value=round(100.0 / n, 2), max_value=100.0, value=100.0, step=0.01)
    
    # Define the average return constraint
    # We calculate the yearly expected return of the portfolio before fixing on minimum
    average_return_daily = returns.mean(axis=0).values # Average return per asset
    w_ratio = np.asarray(weight, dtype=float) / 100.0
    
    daily_return_base = float(average_return_daily @ w_ratio)
    if horizon == '1 Day':
        time_to_year=252
    elif horizon == '1 Week': 
        time_to_year=52
    
    yearly_return_base = (1.0 + daily_return_base)**time_to_year - 1.0  
    c1.write(f"Annual Expected Return of the Initial Portfolio : {yearly_return_base*100:.2f}%")
    
    
    
    min_return_annual_constraint_ptopti = c2.number_input("Annualized Expected Return (%)", 
                            min_value=0.0, max_value=10000.0, value=float(yearly_return_base*100), step=0.1)
    # Computation of the daily return for the constraint
    min_return_daily_pt_opti = (1.0 + min_return_annual_constraint_ptopti/100.0)**(1/time_to_year) - 1.0
    
    # First step : defition of the variables for optimization
    w = cp.Variable(n) # Creation of a vector of size n (the number of stocks given before)
    
    
    losses = -R @ w # It is a matrix product so we use @ instead of *
    # R @ w = vector with the losses, so we put de - to have a positive number for losses
    # Why do we do this ? Because in the rockafellar model, we need to compare every losses with the VaR
    
    # Creation of the variables for the model
    # xi is a scalair product for the optimization
    xi = cp.Variable() # In cvxpy, this function creates a solver (it's our VaR), the solver will find its optimal value for our objective
    
    # u will track which days the loss is over the VaR
    u = cp.Variable(J, nonneg=True) # nonneg = True in order to force it being above 0
    
    # What's the goal here ? Minimize the average of all the times when we're over the VaR (u the CVaR)
    # But we also try to minimize the VaR (xi) otherwise the model will just take a big VaR reducing the risk too much
    # So after the resolution, xi.value will be the optimal VaR, and u.value the excess above the VaR
    # To CONCLUDE: we try to have the minimal number of excess above the VaR (CVaR) while trying to have the lowest VaR
    
    
    # Rockafellar formula: cvar = VaR (xi) + (1/(1-confidence_level/100))*(sum_of_u/nb_of_days)
    cvar_calcul = xi + (1.0 / (1.0 - alpha/100)) * cp.sum(u) / J
    
    
    # Now we put the constraints for the solver
    # We already have u > 0 with the nonneg = True (excesses cannot be negative)
    # Here we put all the constraints like this
    constraints = [ u >= losses - xi, cp.sum(w)==1, w>=0, w <= cap_max /100, 
                   average_return_daily @ w >= min_return_daily_pt_opti,
                   w >= cap_min / 100]
    
    # u >= losses - xi : we only keep the moment the losses are over the VaR basically
    # cp.sum(w)==1 : we just max the allocation of the portfolio at 100%
    # w>= 0 : the solver can't short positions, only long (it's a choice here)
    #  w <= cap/100 : we prevent the solver to "all in"
    # w >= cap_min / 100 : we put a minimum allocation
    
    
    # Now we write the problem
    problem = cp.Problem(cp.Minimize(cvar_calcul), constraints)
    # We instruct cvxpy to minimize our VaR while enforcing our constraints
    
    # Now we solve the prolblem
    solution = problem.solve(solver=cp.SCS, verbose=False)
    # Here we tell it to solve with the solver scs, and the verbose = False hides the logs
    # The variables w, xi, u, will all take their optimized value
    
    # We obtain the optimized allocation as a one-dimensional list
    weight_optimized = np.asarray(w.value).ravel()
    var_optimized = float(xi.value)
    u_optimized = np.asarray(u.value).ravel()
    cvar_optimized = var_optimized + (1.0 / (1.0 - alpha/100.0)) * u_optimized.mean()

    
    daily_return_2 = (1 + returns).cumprod().shift(1).fillna(1.0)
    
    # Computation of the portfolio growth with the optimal weights
    portfolio_optimized = initial_investment * daily_return_2.mul(weight_optimized, axis=1).sum(axis=1) # we sum the weighted values
    portfolio_returns_optimized = portfolio_optimized.pct_change().dropna()
    
    # Recompute the old portfolio perf
    portfolio_base = initial_investment * daily_return_2.mul(w_ratio, axis=1).sum(axis=1)
    portfolio_returns_base = portfolio_base.pct_change().dropna()
    
    
    returns_base = st.session_state["portfolio_returns"]
    
    
    var_pt_base = np.quantile(portfolio_returns_base, 1 - alpha/100)
    cvar_pt_base = portfolio_returns_base[portfolio_returns_base  <= var_pt_base].mean()
    
    var_pt_optimized = np.quantile(portfolio_returns_optimized, 1 - alpha/100)
    cvar_pt_optimized = portfolio_returns_optimized[portfolio_returns_optimized <= var_pt_optimized].mean()
     
    # Recompute the cumulative returns of both portfolios, we divide the last value by the first -1
    growth_pt_base = portfolio_base.iloc[-1] / portfolio_base.iloc[0] - 1
    growth_pt_optimized = portfolio_optimized.iloc[-1] / portfolio_optimized.iloc[0] - 1
    
    # Plotting the Chart and the Weights of the Optimization
    st.write("#### Weights & Returns of the Optimized Portfolio")
    
    # Plotting the Chart of Both Portfolios Returns
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(portfolio_base, color='red',label='Initial Portfolio') 
    ax.plot(portfolio_optimized, color='green',label='Optimized Portfolio') 
    ax.set_title('Growth of Initial & Optimized Portfolio | Rockafellar-Uryasev Optimization')
    ax.legend()
    ax.grid()
    st.pyplot(fig)
    
    weights_optimized = pd.DataFrame({
        "Ticker": tickers,                                 
        "Base (%)":  np.round(w_ratio * 100.0, 2),
        "Optimized (%)": np.round(weight_optimized * 100.0, 2)
    })
    styled_df = (
        weights_optimized.style
        .background_gradient(subset=["Base (%)"], cmap="Reds")
        .background_gradient(subset=["Optimized (%)"], cmap="Greens")
        .format({"Base (%)": "{:.2f}", "Optimized (%)": "{:.2f}"})
        .set_properties(**{
            "text-align": "center",
            "font-weight": "bold",
            "border": "4px solid #ddd",
        })
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.write("### Outputs of the Initial & Optimized Portfolios")

    data = [
    ("VaR", f"${-var_pt_base*initial_investment:,.2f} | {-var_pt_base*100:,.2f}%", f"${-var_pt_optimized*initial_investment:,.2f} | {-var_pt_optimized*100:,.2f}%",
     f"{-(var_pt_base - var_pt_optimized)/abs(var_pt_base)*100:.1f}% better"),
    ("CVaR", f"${-cvar_pt_base*initial_investment:,.2f} | {-cvar_pt_base*100:,.2f}%", f"${-cvar_pt_optimized*initial_investment:,.2f} | {-cvar_pt_optimized*100:,.2f}%",
     f"{-(cvar_pt_base - cvar_pt_optimized)/abs(cvar_pt_base)*100:.1f}% better"),
    ("Growth", f"{growth_pt_base*100:.2f}%", f"{growth_pt_optimized*100:.2f}%",
     f"{(growth_pt_optimized - growth_pt_base)*100:.1f}% ↑"),
]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Metric", "Original", "Optimized", "Δ Improvement"],
            fill_color='#1F618D',
            align='center',
            font=dict(color='black', size=12)
        ),
        cells=dict(
            values=[[row[0] for row in data],
                    [row[1] for row in data],
                    [row[2] for row in data],
                    [row[3] for row in data]],
            fill_color=[['#000000', '#000000']*3],
            align='center',
            height=30)
    )])
    
    fig.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()

    submitted_5 = st.checkbox('Show Efficient Frontier')
    
    if submitted_5:         
        # Generating risk levels in %
        if horizon == '1 Day':
            risk_levels = np.linspace(0.01, 0.3, 100)
        else: 
            risk_levels = np.linspace(0.01, 0.4, 100)            
        # Setting different confidence levels
        alpha_array = [90, 95, 99]
        
        efficient_frontier = {alpha: [] for alpha in alpha_array}

        for risk_level in risk_levels:
            for alpha_s in alpha_array:
        
                # Variables to find
                w = cp.Variable(n)          # Weights
                VaR = cp.Variable()        
                u = cp.Variable(J, nonneg=True)
        
                # Losses
                losses = -R @ w  
        
                # CVaR formula
                cvar = VaR + (1.0 / ((1 - alpha_s/100) * J)) * cp.sum(u)
        
                # Constraints
                constraints = [
                    u >= losses - VaR, 
                    u >= 0,
                    cvar <= risk_level,          # CVaR constraint
                    cp.sum(w) == 1,              # fsum of weights = 1
                    w >= cap_min, # no short position because cap min is always >= 0
                    w <= cap_max / 100
                ]
        
                # Objective: Maximising Expected Return = min(-R(x)) = -min(R(x)) = max(R(x))
                expected_return = cp.sum(R @ w) / J
                objective = cp.Maximize(expected_return)
        
                # Resolving the problem
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.ECOS, verbose=False)
        
                if w.value is not None:
                    weights_opt = np.asarray(w.value).ravel()
                    ret_opt = (R @ weights_opt).mean()
                    efficient_frontier[alpha_s].append({
                        "expected_return": ret_opt*100*time_to_year,
                        "risk": risk_level*100,
                        "weights": weights_opt
                    })
        
        i = 0
        n_points = len(efficient_frontier[90])
        
        while i < n_points:
        
            exp_ret_1 = round(efficient_frontier[90][i]['expected_return'], 1)
            exp_ret_2 = round(efficient_frontier[95][i]['expected_return'], 1)
            exp_ret_3 = round(efficient_frontier[99][i]['expected_return'], 1)
        
            # Break conditions : convergence of the confidence levels
            if exp_ret_1 == exp_ret_2 == exp_ret_3:
                selected_risk = efficient_frontier[90][i]['risk']
                break
        
            i += 1
    
        col1, col2 = st.columns(2)
        # Setting the min and the max return to create an array in order to generate the x-values
        min_return = col1.number_input('Enter the min return', value=efficient_frontier[90][0]['risk'], step=1.0)
        max_return = col2.number_input('Enter the max return', value=efficient_frontier[99][i+1]['risk'], step=1.0)
        
        efficient_frontier = {
            alpha: [pt for pt in efficient_frontier[alpha] if (pt['risk'] <= max_return) & (pt['risk'] >= min_return)]
            for alpha in alpha_array
        }

        # Plotting the Efficient Frontier of the portfolio with Rockafellar Optimization
        fig, ax = plt.subplots()
        for alpha_s, points in efficient_frontier.items():
            x = [p["risk"] for p in points]
            y = [p["expected_return"] for p in points]
            ax.plot(x, y, label=f"alpha={alpha_s}%")
        
        ax.set_xlabel("Risk (in %)")
        ax.set_ylabel("Annual Expected Return (in %)")
        ax.set_title('Efficient Frontiers with Rockafellar-Uryasev')
        ax.grid()
        ax.legend()
        st.pyplot(fig)