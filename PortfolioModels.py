
# ===============================
# Portfolio Optimization Models
# ===============================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import statsmodels.api as sm
import warnings
warnings.simplefilter("ignore")

# ----------------------------------------------------------------------

# Modelos de Markowitz y Sharpe (Media-Varianza - MV) 

# Portafolio de Minima Varianza global de Markowitz (PMVg)

def markowitzMV(retornos):
    mu = retornos.mean()         
    cov = retornos.cov()         
    n = len(mu)            
    wi = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(wi,cov))
    constraints = [ wi.sum() == 1, wi >= 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    wpo = (wi.value).round(6) 
    return wpo

# Portafolio de Minima Varianza de Markowitz con Retorno Objetivo: Portafolios de la FE

def markowitzFE(retornos, rpo):
    mu = retornos.mean()         # Retornos esperados
    cov = retornos.cov()         # Covarianzas
    n = len(mu)                  # No. de activos
    wi = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(wi,cov))
    constraints = [ wi.sum() == 1, wi >= 0, wi @ mu == rpo ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    wpo = (wi.value).round(6) 
    return wpo


# Portafolio Tangente de Sharpe

def sharpeMV(retornos):
    mu = retornos.mean()         # Retornos esperados
    cov = retornos.cov()         # Covarianzas
    n = len(mu)                  # No. de activos
    wi = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(wi,cov))
    constraints = [ wi >= 0, wi @ mu == 1]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    wpo = (wi.value/wi.value.sum()).round(6) 
    return wpo

# ----------------------------------------------------------------------

# Sortino Media-SemiVarianza (MsV)

# Portafolio de Minimo Riesgo (Semivarianza) de Sortino

def sortinoMR(retornos,h):
    mu = retornos.mean() 
    semiretornos = np.minimum(retornos, h)
    semicov = semiretornos.cov()               
    n = len(mu)            
    wi = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(wi,semicov))
    constraints = [ wi.sum() == 1, wi >= 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    wpo = (wi.value).round(6) 
    return wpo

# Portafolio de Minima Semi-Varianza de Sortino con Retorno Objetivo: Portafolios de la FE

def sortinoFE(retornos, rpo, h):
    mu = retornos.mean() 
    semiretornos = np.minimum(retornos, h)
    semicov = semiretornos.cov()         
    n = len(mu)                  
    wi = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(wi,semicov))
    constraints = [ wi.sum() == 1, wi >= 0, wi @ mu == rpo ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    wpo = (wi.value).round(6) 
    return wpo


# Portafolio Tangente de Sortino

def sortinoT(retornos,h):
    mu = retornos.mean() 
    semiretornos = np.minimum(retornos, h)
    semicov = semiretornos.cov()         
    n = len(mu)                    
    wi = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(wi,semicov))
    constraints = [ wi >= 0, wi @ mu == 1]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    wpo = (wi.value/wi.value.sum()).round(6) 
    return wpo


# ----------------------------------------------------------------------

# Modelo de Treynor

def treynor(retornos, rindice):
    mu = retornos.mean()
    n = len(mu)
    X = sm.add_constant(rindice)
    alphas = []
    betas = []
    rcuadrados = []
    varerrores = []
    for i in range(n):
        y = retornos.iloc[:,i]
        modelo = sm.OLS(y,X).fit()
        alpha = float(modelo.params[0])#.item()
        beta = float(modelo.params[1])#.item()
        rcuadrado = modelo.rsquared_adj
        varerror = modelo.resid.var()
        betas.append(beta)
        alphas.append(alpha)
        rcuadrados.append(rcuadrado)
        varerrores.append(varerror)
    #nombres = mu.index.to_list()
    treynori = mu/betas
    medidas = pd.DataFrame({'mu': mu, 'alphas':alphas,'betas':betas,'R^2':rcuadrados,'Sigma^2':varerrores,'Treynori':mu/betas})
    medidas = medidas.sort_values(by=['Treynori'],ascending=False)
    rf = 0
    ratio1 = (medidas['mu'] - rf ) * medidas['betas'] / medidas['Sigma^2']
    ratio2 = medidas['betas']**2 / medidas['Sigma^2']
    suma1 = np.cumsum(ratio1)
    suma2 = np.cumsum(ratio2)
    sigmam2 = float(rindice.var()[0])#.item()
    tasac = sigmam2 * suma1 / (1+sigmam2*suma2)
    zi = medidas['betas'] / medidas['Sigma^2'] * (medidas['Treynori'] - np.max(tasac))
    wpo = (np.maximum(zi,0))/(np.maximum(zi,0)).sum()  
    return wpo

# ----------------------------------------------------------------------

# Modelo Omega

#Portafolio de Minimo Riesgo Omega

def omegaMR(retornos,h):
    mu = retornos.mean() 
    n = len(mu)      
    wi = cp.Variable(n)
    rp = retornos.to_numpy() @ wi
    objective = cp.Minimize( -cp.minimum(rp,h).sum() )
    constraints = [ wi.sum()== 1, wi >= 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    wpo = (wi.value).round(6)
    return wpo

# Portafolio Omega con Retorno Objetivo: Portafolios de la FE

def omegaFE(retornos,rpo,h):
    mu = retornos.mean() 
    n = len(mu)      
    wi = cp.Variable(n)
    rp = retornos.to_numpy() @ wi
    objective = cp.Minimize( -cp.minimum(rp,h).sum() )
    constraints = [ wi.sum()== 1, wi >= 0, wi @ mu == rpo ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    wpo = (wi.value).round(6)
    return wpo

# Portafolio Omega Tangente

def omegaT(retornos,h):
    mu = retornos.mean() 
    n = len(mu)      
    wi = cp.Variable(n)
    rp = retornos.to_numpy() @ wi
    objective = cp.Minimize( -cp.minimum(rp,h).sum() )
    constraints = [ wi >= 0, wi @ mu == 1 ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    wpo = (wi.value/wi.value.sum()).round(6)
    return wpo


# ----------------------------------------------------------------------

# Modelo Media-CVaR

#Portafolio de Minimo Riesgo CVaR

def cvarMR(retornos,alpha):
    t = len(retornos)
    mu = retornos.mean() 
    n = len(mu) 
    beta = 1-alpha
    wi = cp.Variable(n)
    portfolio_return = retornos.values @ wi
    VaR = cp.Variable()
    cvar = VaR + (1 / (1 - beta)) * cp.sum(cp.pos(-portfolio_return - VaR))/t
    objective = cp.Minimize(cvar)                    # Minimizar CVaR
    constraints = [wi.sum() == 1, wi >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()                                  # GLPK method
    wpo = wi.value
    return wpo.round(6)


# Portafolio CVaR con Retorno Objetivo: Portafolios de la FE

def cvarFE(retornos,rpo,alpha):
    t = len(retornos)
    mu = retornos.mean() 
    n = len(mu) 
    beta = 1-alpha
    wi = cp.Variable(n)
    portfolio_return = retornos.values @ wi
    VaR = cp.Variable()
    cvar = VaR + (1 / (1 - beta)) * cp.sum(cp.pos(-portfolio_return - VaR))/t
    objective = cp.Minimize(cvar)  
    constraints = [wi >= 0, wi.sum() == 1, portfolio_return.mean()==rpo]
    problem = cp.Problem(objective, constraints)
    problem.solve() 
    wpo = wi.value
    return wpo.round(6)

# Portafolio Media-CVaR Tangente

def cvarT(retornos,alpha):
    t = len(retornos)
    mu = retornos.mean() 
    n = len(mu) 
    beta = 1-alpha
    wi = cp.Variable(n)
    portfolio_return = retornos.values @ wi
    VaR = cp.Variable()
    cvar = VaR + (1 / (1 - beta)) * cp.sum(cp.pos(-portfolio_return - VaR))/t
    objective = cp.Minimize(cvar)  
    constraints = [wi >= 0, portfolio_return.mean()==1]
    problem = cp.Problem(objective, constraints)
    problem.solve() 
    wpo = wi.value / wi.value.sum()
    return wpo.round(6)

# =======================================================================

# Perfomance

def performance(retornos,wpo):
    rhpo = retornos @ wpo
    t = len(rhpo)
    valorpo = np.zeros(t+1) 
    valorpo[0] = 1 
    for i in range(1, t+1):
        valorpo[i] = valorpo[i-1] * np.exp(rhpo.iloc[i-1])
    return valorpo

def medidas(retornos,rindice,wpo):
  rph = retornos @ wpo
  rp = rph.mean().item()
  sigmap = rph.std()
  sharpep = rp/sigmap
  semirph = np.minimum(rph,0)
  sortinop = rp/semirph.std()
  modelo = sm.OLS(rph,sm.add_constant(rindice)).fit()
  betap = modelo.params[1].item()
  treynorp = rp/betap
  omegap = np.maximum(rph,0).sum()/-np.minimum(rph,0).sum()
  varp = np.percentile(rph,5)
  cvarp = rph[rph < varp].mean().item()
medidas = {'Retorno': rp, 'Volatilidad':sigmap, 'Sharpe': sharpep, 'Beta':betap,'Treynor':treynorp,'Sortino': sortinop,
             'Omega': omegap.item(), 'CVaR 95%': cvarp }}
  return medidas
