import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Importing tickers from a csv
tickers_1 = pd.read_csv("tickers.csv", sep=";") 
tickers_1 = tickers_1.set_index('Symbol')

# Function to import data from Yahoo Finance
def importing_price(tickers, time_horizon, years: int = 5):
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
            # We rename the closing series
            close_series = raw["Close"]
            close_series.name = ticker
            all_data.append(close_series)
        else:
            st.warning(f"No data for {ticker}")

        time.sleep(0.5)  # Small pause to avoid rate-limit

    # Concatenate
    if all_data:
        data = pd.concat(all_data, axis=1)

        # Temporary common base
        end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        start_date = end_date - pd.DateOffset(years=years)
        date_index = pd.date_range(start=start_date, end=end_date, freq='B')

        # Reindexation and cleaning
        data = data.reindex(date_index)
        data = data.dropna(how="all")
        return data
    else:
        return pd.DataFrame()
        

# User interform 
st.title('📊 VaR & CVaR Models')

st.caption("This page allows you to choose a model to simulate VaR and CVaR")

# Risk Metric Inputs for Principal Parameters (VaR and CVaR)
st.write('### Principal Parameters')
col1, col2 = st.columns(2)

# Mapping horizon labels to trading days
horizon_to_days = {
    '1 Day': '1d',
    '1 Week': '5d'
}

day_horizon = {
    '1 Day': 1,
    '1 Week': 5
}

# Choosing the horizon
h_array = ['1 Day', '1 Week']
horizon = col1.selectbox('Select your Time Horizon', h_array)
time_horizon = horizon_to_days.get(horizon)
trading_days = day_horizon.get(horizon)

alpha = float(col2.number_input('Enter your Confidence Level (in %) | MAX 99.97%', value = 99.0, step=1.0, max_value=99.97))

# Choosing your assets
st.write('### Assets & Initial Portfolio Value')
col1, col2 = st.columns(2)

n = int(col1.text_input('Number of assets', value = 2))
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
 
# Initialising the state
if "submitted_1" not in st.session_state:
    st.session_state["submitted_1"] = False

# Button 
submitted_1 = st.button("Choose your VaR Model")

if submitted_1:
    st.session_state["submitted_1"] = True  # turning the state into true


if st.session_state["submitted_1"]:
    total = sum(weight)
    
    if total == 100.0:
        # Importing the data
        data = importing_price(tickers, time_horizon, 5) 
        data = data.loc[:, tickers] # Ordering the tickers according the choice made before
        data_clean = data.dropna(how="any") # All the assets need to have a price to be kept in the data
        
        # Calculate the returns (convert the price in changes %) without taking the NA
        returns = data_clean.pct_change().dropna(how="any")
        w = np.asarray(weight, dtype=float) / 100.0
        w = w / w.sum()
            
        # Print the returns of the portfolio "Buy & Hold" style
        assets_w = (1.0 + returns).cumprod()                  
        portfolio_value = initial_investment * (assets_w @ w)
        portfolio_returns = portfolio_value.pct_change().dropna()
        
        # Stock the data
        st.session_state["portfolio_returns"] = portfolio_returns
        st.session_state["returns"] = returns
        st.session_state["weight"] = weight
        

        # Historical Value at Risk modelization
        st.divider()
        hist_var = st.checkbox("Historical Value at Risk")

        if hist_var: 
            # Getting the former data
            portfolio_returns = st.session_state["portfolio_returns"]
            
            # Calculating the VaR
            var_level = np.quantile(portfolio_returns, 1 - alpha/100)
            
            # Calculating the CVaR
            cvar_level = portfolio_returns[portfolio_returns <= var_level].mean()
            
            # Listing all tickers with Portfolio
            tickers_2 = ['PORTFOLIO'] + tickers
            
            st.write("### Plotting the Asset / Portfolio Distribution")
            # Showing the histogram of each assets and portfolio
            hist_select = st.selectbox('Select your Distribution', tickers_2)
            
            fig, ax = plt.subplots()
            if hist_select in tickers:
                # Data from the sticker
                data_sticker = returns[hist_select]
                mean = data_sticker.mean()
                std = data_sticker.std()
                
                # Generating the plot
                sns.histplot(data_sticker, kde=True, stat='density', bins=40)
                
                # Normal Distribution
                x = np.linspace(mean - 3*std, mean + 3*std, 250)
                y = norm.pdf(x, mean, std)
                ax.plot(x, y, color='red', lw=2, label='Normal Distribution')
            else: 
                fig, ax = plt.subplots()
                
                # Data from the sticker
                data_sticker = portfolio_returns
                mean = data_sticker.mean()
                std = data_sticker.std()
                
                # Generating the plot
                sns.histplot(data_sticker, kde=True, stat='density', bins=40)
                
                # Normal Distribution
                x = np.linspace(mean - 3*std, mean + 3*std, 250)
                y = norm.pdf(x, mean, std)
                ax.plot(x, y, color='red', lw=2, label='Normal Distribution')
                ax.axvline(var_level, color='black', linestyle='--', lw=2, label=f'Value at Risk at {alpha}%')
            
            ax.set_xlabel(f"Returns for {hist_select} (in %)")
            ax.set_title(f"Histogram of {hist_select}")
            ax.legend()
            st.pyplot(fig)
            
            st.write("### VaR and CVaR Outputs")
            col1, col2 = st.columns(2)
            # VaR Outputs
            col1.write(f"Historical VaR (in %): {-var_level*100:.2f}%")
            col2.write(f"Historical VaR (in USD): ${-var_level*initial_investment:.2f}")
            
            # CVaR Outputs
            col1.write(f"Historical CVaR (in %): {-cvar_level*100:.2f}%")
            col2.write(f"Historical CVaR (in USD): ${-cvar_level*initial_investment:.2f}")        
        
        st.divider()
        mv_var = st.checkbox("Variance Covariance Value at Risk")
        
        if mv_var: 
            # Calculating the Cov Matrix
            corr_matrix = returns.corr()
            
            st.write("### Heatmap of the Correlation between the Assets")
            
            # Plotting the Correlation Matrix between assets
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.heatmap(corr_matrix, annot=True)
            
            ax.set_title('Heatmap - Correlation Matrix')
            
            st.pyplot(fig) 
            
            # Calculating the mean of the portfolio
            mean_returns = returns.mean().to_numpy() @ w
            
            # Calculating the Covariance Matrix
            cov_matrix = returns.cov()
            
            # Calculating the Portfolio Standard Deviation
            weight_array = w 
            variance_portfolio = weight_array @ cov_matrix @ weight_array.T
            std_portfolio = np.sqrt(variance_portfolio)
            
            # Calculating the z-value
            z_value = norm.ppf(alpha/100)
            
            # Calculating the VaR
            var = (mean_returns - (std_portfolio * z_value))
            
            # Calculating the CVaR
            cvar_level = portfolio_returns[portfolio_returns <= var].mean()

            # Plotting the VaR and the portfolio distributions
            st.write('### Plotting the Portfolio Distribution')
            fig, ax = plt.subplots(figsize=(6,4))
                
            # Data from the sticker
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            
            # Generating the plot
            sns.histplot(portfolio_returns, kde=True, stat='density', bins=40)
            
            # Normal Distribution
            x = np.linspace(mean - 3*std, mean + 3*std, 250)
            y = norm.pdf(x, mean, std)
            ax.plot(x, y, color='red', lw=2, label='Normal Distribution')
            ax.axvline(var, color='black', linestyle='--', lw=2, label=f'Value at Risk at {alpha}%')
            
            ax.set_xlabel("Returns for PORTFOLIO (in %)")
            ax.set_title("Histogram of PORTFOLIO")
            ax.legend()
            st.pyplot(fig)
            
            st.write("### VaR and CVaR Outputs")
            col1, col2 = st.columns(2)
            # VaR Outputs
            col1.write(f"Variance Covariance VaR (in %): {-var*100:.2f}%")
            col2.write(f"Variance Covariance VaR (in USD): ${-var*initial_investment:.2f}")
            
            # # CVaR Outputs
            col1.write(f"Variance Covariance CVaR (in %): {-cvar_level*100:.2f}%")
            col2.write(f"Variance Covariance CVaR (in USD): ${-cvar_level*initial_investment:.2f}") 
            
        
        # Monte-Carlo Value at Risk modelization
        st.divider()
        montecarlovar = st.checkbox('Monte Carlo Value at Risk')
        
        if montecarlovar: 
            # Separating the Monte Carlo into two possible modelizations
            type_montecarlo = st.selectbox('Select your Monte Carlo Model',['Stationary Monte Carlo Model', 'Hidden Markov Model'])
            
            # Batching once again just to have daily simulations instead of weekly ones
            if horizon != '1 Day':
                # Importing the data
                data = importing_price(tickers, '1d', 5) 
                data = data.loc[:, tickers] # Ordering the tickers according the choice made before
                data_clean = data.dropna(how="any") # All the asset need to have a price to keep the data
                
                # Calculate the returns (convert the price in changes %) without taking the NA
                returns = data_clean.pct_change().dropna(how="any")
            
            # Calculating the cov matrix
            corr_matrix = returns.corr()
            
            # Heatmap of the Covariance Matrix
            st.write('### Heatmap of the Correlation between the Assets')
            
            # Plotting the Correlation Matrix between assets
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.heatmap(corr_matrix, annot=True)
            
            ax.set_title('Heatmap - Correlation Matrix')
            
            st.pyplot(fig) 
            
            # Monte Carlo Simulation on a 2D Plot
            st.write("### Monte Carlo Simulation on a 2D Plot")
            # Inputs for the Monte Carlo Simulation
            col1, col2 = st.columns(2)
            
            nb_simulations = col1.number_input('Number of Simulations', min_value = 200, max_value = 1000, value = 600, step=100, key="nb_simulations_mc")
            time_range = col2.number_input('Number of Trading Days', min_value = 1, max_value = 500, value = trading_days, step=25, key="time_range_mc")
            
            # STATIONARY MONTE CARLO 
            if type_montecarlo == 'Stationary Monte Carlo Model':
                # Calculate the mean of returns 
                mean_returns = returns.mean()
                
                # Calculate the cov matrix
                cov_matrix = returns.cov()
                
                # Storage for information
                mean_matrix = np.tile(mean_returns.values, (time_range, 1)) 
                
                
                # Store each iteration
                portfolio_simulations_mc = np.full(shape=(time_range+1, nb_simulations), fill_value=0.0) #we put nothing in it
                portfolio_simulations_mc[0, :] = initial_investment
                
                
                # Allocation before the loop to simulate Buy & Hold
                allocations = initial_investment * w
                
                D = returns.shape[1]
                # Creating the lower triangle matrix with the Cholesky decomposition (because in real life, assets are not independent, they have a covariance)
                lower_triangle = np.linalg.cholesky(cov_matrix.values)
    
                for sim in range(0, nb_simulations):
                    
                    
                    # Using the Cholesky decomposition here to determine the lower triangular matrix
                
                    # Creating a random number following the normal distribution for each return of the assets
                    matrice_rnd = np.random.normal(size=(time_range, D))
                    
                    
                    # Calculate the daily returns
                    daily_returns = mean_matrix + matrice_rnd @ lower_triangle.T
                    
                    cum_returns = np.cumprod(daily_returns + 1, axis=0)
                    cum_returns = np.vstack([np.ones((1, D)), cum_returns])
                    
                    value_per_asset = allocations * cum_returns
                    portfolio_simulations_mc[:, sim] = value_per_asset.sum(axis=1)
    
                portfolio_results_mc = pd.Series(portfolio_simulations_mc[-1, :])
                
                 
                def mc_var(returns, alpha=5):
                    return np.percentile(returns, alpha)
                
                def mc_cvar(returns, alpha=5):
                    below = returns <= mc_var(returns, alpha=alpha)
                    return returns[below].mean() # As usual, cvar is the mean of returns below the var
                
                
                var_mc   = initial_investment - mc_var(portfolio_results_mc,  alpha=100.0-alpha)
                cvar_mc  = initial_investment - mc_cvar(portfolio_results_mc,  alpha=100.0-alpha)
                
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(portfolio_simulations_mc, alpha = 0.5) # We trace the simulations
                ax.axhline(initial_investment, color='black', linestyle='--', label='Initial Investment')
                ax.axhline(initial_investment - var_mc, color='red', linestyle='--', label=f'Value at Risk at {alpha}%')
                ax.legend()
                ax.set_title(f'Monte Carlo Portfolio Simulation after {time_range} days')
                ax.set_xlabel("Number of Days")
                ax.set_ylabel("Portfolio Value")
                st.pyplot(fig) #show it
                
                # Monte Carlo Distribution
                st.write("### Monte Carlo Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot((portfolio_results_mc - initial_investment)/initial_investment, bins=25, kde=True)
                ax.axvline(- var_mc/initial_investment, color='red', linestyle='--', label=f'Value at Risk at {alpha}%')
                ax.legend()
                ax.set_title('Monte Carlo Distribution')
                st.pyplot(fig) #show it
                
                
                # Output for the Monte Carlo Value at Risk
                st.write("### VaR and CVaR Outputs")
                
                col1, col2 = st.columns(2)
                
                col1.write(f"Monte Carlo VaR (in %): {var_mc/initial_investment*100:.2f}%")
                col2.write(f"Monte Carlo VaR (in USD): ${var_mc:.2f}")
                
                col1.write(f"Monte Carlo CVaR (in %): {cvar_mc/initial_investment*100:.2f}%")
                col2.write(f"Monte Carlo CVaR (in USD): ${cvar_mc:.2f}")
            
            else:
                # Get the likelihood that returns of the day are in  state k, gives us scores of likeliness
                def log_gauss(returns, means, covariances):
                    
                    T, assets_nb = returns.shape 
                    hidden_states_nb = means.shape[0] # we take the number of lines
                    
                    log_likelihood = np.empty((T, hidden_states_nb)) # creation of an empty matrice of required size
                    # Will contain the likelihood of day t under state k
                    
                    for k in range(hidden_states_nb):
                        
                        matrix_cov = covariances[k] # Cov matrix in state k
                        
                        L = np.linalg.cholesky(matrix_cov) # More stable for covariance
                        
                        difference = returns - means[k] # Get the deviation of day t in regime k
                        
                        gd = np.linalg.solve(L, difference.T) # Calculate the gaussian density
                        
                        # We calculate the distance of Mahalanobis, it indicates how distant the return of day t is from the center
                        distmaha = np.sum(gd*gd, axis=0) 
                        
                        logdet = 2.0 * np.sum(np.log(np.diag(L)))
                        
                        log_likelihood[:, k] = -0.5 * (assets_nb * np.log(2 * np.pi) + logdet + distmaha)
                        
                    return log_likelihood
                         
    
                def forward_log(log_emlik, log_pi, log_A):
                    
                    # log_emlik : log P(observations t for state k) -> the output of the log_gauss fonction
                    # log_pi : log initial proba
                    # log transition matrix
                        
                    T, K = log_emlik.shape # T = nb of days / K = nb of hidden states
                    
                    # Will contain the log probability of having observed everything to day t, and being in state k
                    log_alpha = np.empty((T,K), dtype=float)
                    
                    # Initialize for the first day
                    # Apply the rule of joint probabilities in log : P(obs0, state0 = k) = P(state0 = k) * P(osb0 | state0 = k)
                    # The multiplication becomes an addition in log
                    # We sum the probability of being in any state at the beginning with the likeliness of the first observation in each state
                    log_alpha[0] = log_pi + log_emlik[0]
                    
                    # We're gonna calculate log_alpha for each day t with the previous day t-1
                    for t in range(1,T):
                        
                        # We add (in log so multiply in reality) the probability of being in the state of the last day (t-1) with the transition matrix
                        # tmatrix gives us a matrix with all the possible individual roads to be today in state k
                        tmatrix = log_alpha[t-1][:, None] + log_A
                        
                        # We do the log-sum-exp trick, our big negative values in log, composed directly by exp, could give us underflows (informatic zeros)
                        # So we substract the max of each column before calculating the exponential
                        # Trick of numerical stability
                        m = np.max(tmatrix, axis=0)
                        
                        # Sum all the roads to be in state k today
                        logsum = np.log(np.sum(np.exp(tmatrix - m), axis=0)) + m
                        
                        # We add (multiply in reality because of log) the probability of observing the returns of day t if we are in the state k to the probability to arrive in state k today
                        log_alpha[t] = log_emlik[t] + logsum
                        
                    return log_alpha
                        
                        
                def backward_log(log_emlik, log_A):
                    
                    T, K = log_emlik.shape
                    
                    log_beta = np.empty((T, K), dtype=float)
                    
                    # Initialisation of the last line at 0, not the same as the forward, because the probability at the end is 1 and log(1)=0
                    log_beta[-1] = 0.0
                        
                    # We do the forward in reverse
                    for t in range(T - 2, -1, -1):
                        
                        # log_A log transition matrix between hidden states
                        # log_emlik[t+1] log prob to observ the returns of the following daw
                        # log_beta[t+1] log prob of everything that happens after
                        tmatrix = log_A + log_emlik[t + 1][None, :] + log_beta[t + 1][:, None]
                    
                        m = np.max(tmatrix, axis=1)
                        
                        # Calculate the sum of all the possible transitions to the future for each current state k
                        logsum = np.log(np.sum(np.exp(tmatrix - m[:, None]), axis=1)) + m
                    
                        log_beta[t] = logsum
                        
                    return log_beta
    
                # E STEP fusion forward backward
    
                def e_step_fusion(log_alpha, log_beta, log_emlik, log_A):
                    
                    T, K = log_alpha.shape
    
                    # Sum because of log, multiply foward and backward, probability to be in each state considering past and future
                    log_gamma = log_alpha + log_beta
                    
                    m = np.max(log_gamma, axis=1)
                    
                    # We normalize the log_gamma to have real probabilities
                    # Before it's a joint probability, not conditional
                    # We have before, what's the proba to be in state k AND that we have all the past and future observations
                    # We want what's the probability to be in state k IF we have all the past and future observations
                    log_gamma = log_gamma - (np.log(np.sum(np.exp(log_gamma - m[:, None]), axis=1)) + m)[:, None]
                    gamma = np.exp(log_gamma)
    
                    # We create a table that will contain all the log probabilities for all the possible transitions between states
                    log_xi = np.empty((T - 1, K, K), dtype=float)
                    
                    for t in range(T - 1):
                        
                        # log P(etat_t = i, etat_t+1 = j, all observations)
                        # We combine all the factors
                        # Past till t, transition between i and j, future observation and future after t+1
                        tmatrix = log_alpha[t][:, None] + log_A + log_emlik[t + 1][None, :] + log_beta[t + 1][None, :]
                        
                        
                        M = np.max(tmatrix)
                        
                        log_xi[t] = tmatrix - (np.log(np.sum(np.exp(tmatrix - M))) + M)
                        
                    xi = np.exp(log_xi)
                    
                    
                    # gamma is the probability for each state at each date
                    # xi is the probability of transition between states
                    return gamma, xi, log_gamma, log_xi
    
                def m_step(returns, gamma, xi, diagonal_shrink=1e-6):
                    
                    T, D = returns.shape
                    
                    K = gamma.shape[1] # Number of hidden states
                    
                    pi_new = gamma[0] # Best estimation for the initial states
                    
                    # Sum matrix of the average transitions
                    A_num = xi.sum(axis=0)
                    
                    # Now we calculate the denominator before the division for the normalization
                    A_den = gamma[:-1].sum(axis=0)[:, None]
                    
                    # We normalize
                    A_new = A_num / A_den
                    
                    # We prepare the new matrices for each state
                    means_new = np.zeros((K, D))
                    covs_new = np.zeros((K, D, D)) # Matrix D*D for each state K
                    
                    for k in range(K):
                        
                        # w contains the temporary weights for the state k at each state t
                        g = gamma[:, k][:, None]
                        
                        # We calculate the total probability of being in state k during the whole period
                        s = g.sum()
                        
                        # We calculate the new mean returns for the state k
                        mu = (g * returns).sum(axis=0) / s
                        means_new[k] = mu # We save the  mean of returns for state k
                        
                        # Calculate the difference between each observations and the means of returns k
                        difference = returns - mu
                        
                        # We calculate the new covariance matrix for the state k
                        covs_new[k] = (difference.T @ (g * difference)) / s + np.eye(D) * diagonal_shrink
                        # diagonal_shrink adds a little term on the diagonal for numerical stability
                        
                    return pi_new, A_new, means_new, covs_new
    
    
                def baum_welch(returns, K, max_iter=50, tol=1e-4, diagonal_shrink=1e-6):
                    # If the model evolves by less than tol, the tolerance, we consider it has converged, we stop the iterations
    
                    rng = np.random.default_rng()
                    
                    T, D = returns.shape
    
                    # Initialize the initial probability vector for hidden states
                    pi = np.full(K, 1.0 / K)
                    
                    A = np.full((K, K), 1.0 / K)  
                    
                    # We initialize the means by state by choosing K random days in historical data
                    means = returns[rng.choice(T, size=K, replace=False)] # False so we don't choose twice the same day
                    
                    # Initialize the covariance matrixes for each hidden state
                    # We suppose that at the beginning all hidden states have the same probability
                    covs = np.array([np.cov(returns.T) + np.eye(D)*1e-6 for _ in range(K)])
                    
                    # Initialize the log likeliness
                    prev_ll = -np.inf
                    
                    
                    for i in range(max_iter):
                        
                        # We calculate the log likeliness
                        log_likeli = log_gauss(returns, means, covs)
                        
                        # Forward
                        log_alpha = forward_log(log_likeli, np.log(pi), np.log(A))
                        
                        # Backward
                        log_beta  = backward_log(log_likeli, np.log(A))
                        
                        M = np.max(log_alpha[-1])
                        
                        # Total likeliness of the current model
                        ll = np.log(np.sum(np.exp(log_alpha[-1] - M))) + M
                        
                        # E step
                        gamma, xi, _, _ = e_step_fusion(log_alpha, log_beta, log_likeli, np.log(A))
                        
                        # M step
                        pi, A, means, covs = m_step(returns, gamma, xi, diagonal_shrink=diagonal_shrink)
                        
                        # Break once it seems accurate
                        if ll - prev_ll < tol:
                            break
                        
                        prev_ll = ll
    
                    return pi, A, means, covs
                
                # Data
                corr_matrix_hmm = returns.corr()
                
                # Monte Carlo Simulation on a 2D Plot
                st.write("### Monte Carlo Simulation under HMM on a 2D Plot")
                # Inputs for the Monte Carlo Simulation
                col1, col2 = st.columns(2)
                
                # Store each iteration
                portfolio_simulations_matrix_hmm = np.full(shape=(time_range+1, nb_simulations), fill_value=0.0) #we put nothing in it
                portfolio_simulations_matrix_hmm[0, :] = initial_investment
                
                
                # Allocation before the loop to simulate buy and hold
                allocations = initial_investment * w
                
                
                # HMM init
                K = 3 # number of hidden states
                
                # Get the variables for the HMM
                pi, A, means, covs = baum_welch(returns.values, K, max_iter=50, tol=1e-4, diagonal_shrink=1e-6) 
                
                # Normalize the lines of A, just a security for the underflows
                A = A / A.sum(axis=1, keepdims=True)
                
                rng = np.random.default_rng() # Rng generator
                D = returns.shape[1]  # Number of assets
                
                
                
                for sim in range(0, nb_simulations):
                    
    
                    state = rng.choice(K, p=pi) # Initial state pi
                    cum_returns = np.ones(D, dtype=float) # Return by asset buy & hold
                
                    for t in range(1, time_range+1):
                        
                        # Generate the rng according to the distribution of the state
                        r_hmm = rng.multivariate_normal(mean=means[state], cov=covs[state])
                        
                        # Returns buy & hold
                        cum_returns = cum_returns*(1.0 + r_hmm)
                        
                        # Portfolio value
                        portfolio_simulations_matrix_hmm[t, sim] = np.dot(allocations, cum_returns)
                        
                        # State update accorging to the transition matrix
                        state = rng.choice(K, p=A[state])
                
                
                
                def mchmm_var(returns, alpha=5):
                    return np.percentile(returns, alpha)
                
                def mchmm_cvar(returns, alpha=5):
                    below = returns <= mchmm_var(returns, alpha=alpha)
                    return returns[below].mean() # As usual, cvar is the mean of returns below the var
                
                
                portfolio_results_hmm = pd.Series(portfolio_simulations_matrix_hmm[-1, :])
                
                
                var_mchmm   = initial_investment - mchmm_var(portfolio_results_hmm,  alpha=100.0-alpha)
                cvar_mchmm  = initial_investment - mchmm_cvar(portfolio_results_hmm,  alpha=100.0-alpha)
                
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(portfolio_simulations_matrix_hmm, alpha = 0.5) # We draw the simulations
                ax.axhline(initial_investment, color='black', linestyle='--', label='Initial Investment')
                ax.axhline(initial_investment - var_mchmm, color='red', linestyle='--', label=f'Value at Risk at {alpha}%')
                ax.legend()
                ax.set_title(f'Monte Carlo Portfolio Simulation under HMM after {time_range} days')
                ax.set_xlabel("Number of Days")
                ax.set_ylabel("Portfolio Value")
                st.pyplot(fig) #show it
                
                # Monte Carlo Distribution
                st.write("### Monte Carlo Distribution under HMM")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot((portfolio_results_hmm - initial_investment)/initial_investment, bins=25, kde=True)
                ax.axvline(- var_mchmm/initial_investment, color='red', linestyle='--', label=f'Value at Risk at {alpha}%')
                ax.legend()
                ax.set_title('Monte Carlo Distribution under HMM')
                st.pyplot(fig) #show it
                
                
                # Output for the Monte Carlo Value at Risk
                st.write("### VaR and CVaR Outputs under HMM")
                
                col1, col2 = st.columns(2)
                
                col1.write(f"Monte Carlo HMM VaR (in %): {var_mchmm/initial_investment*100:.2f}%")
                col2.write(f"Monte Carlo HMM VaR (in USD): ${var_mchmm:.2f}")
                
                col1.write(f"Monte Carlo HMM CVaR under  (in %): {cvar_mchmm/initial_investment*100:.2f}%")
                col2.write(f"Monte Carlo HMM CVaR (in USD): ${cvar_mchmm:.2f}")
                
