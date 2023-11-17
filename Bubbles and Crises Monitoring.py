# Databricks notebook source
!pip install --upgrade pip
!pip install rpy2==3.4.5


# COMMAND ----------

import requests
from base64 import b64encode
import pandas as pd
import re
%load_ext rpy2.ipython
from rpy2.robjects.packages import importr
#import rpy2

import warnings
warnings.filterwarnings('ignore')


# COMMAND ----------

# MAGIC %md
# MAGIC ###Extração dos Dados

# COMMAND ----------

import base64

def codificar(nome_empresa):
    texto = f'{{"language":"pt-br","pageNum":1,"pageSize":20,"tradingName":"{nome_empresa}"}}'
    codigo_base64 = base64.b64encode(texto.encode()).decode()
    return codigo_base64    



# COMMAND ----------

def proventos (nome_empresa):


    string	= codificar(nome_empresa)

    r = requests.get('https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/' +
    string)

    return r.json()


# COMMAND ----------

def get_names():
  get_name = requests.get(r'https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMCwiaW5kZXgiOiJJQk9WIiwic2VnbWVudCI6IjEifQ==')
  dados = get_name.json()['results']
  names = [d['asset'] for d in dados]

  return names






# COMMAND ----------

def get_dataframes():
  names = get_names()
  list_dataframes = []
  for i in names:
      list_dataframes.append(pd.DataFrame(proventos(i)['results']))

  return list_dataframes



# COMMAND ----------

# MAGIC %md
# MAGIC ###Transformação dos Dados

# COMMAND ----------

def get_df():
  df = pd.DataFrame()
  list_dataframes =  get_dataframes()
  for i in list_dataframes:
      df = pd.concat([df, i])
  return  df

# COMMAND ----------

def slice_df():
  df = get_df()
  df_dividendo = df.query('corporateAction == "DIVIDENDO"')
  df_capital = df.query('corporateAction == "JRS CAP PROPRIO"')
  df_total = pd.concat([df_dividendo, df_capital])
  return df_dividendo, df_capital, df_total

# COMMAND ----------

df_dividendo, df_capital, df_total = slice_df()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Refinamento e Exportação dos Dados

# COMMAND ----------

import pandas as pd
class Dataframe(object):

  def __init__(self, df, name):
    self.name = name
    self.df = df

  def slice (self):
    self.df = self.df[['dateApproval' , 'corporateActionPrice']]

  def drop(self):
    self.df.dropna(inplace=True)

  def transform_date(self):
    self.df['dateApproval'] = pd.to_datetime(self.df['dateApproval'], format='%d/%m/%Y')

  def sort(self):
    self.df = self.df.sort_values(by=['dateApproval'], ascending=False)


  def replace(self):
    self.df['corporateActionPrice'] = self.df.loc[:,'corporateActionPrice'].apply(lambda x: float(x.split()[0].replace(',', '.')))


  def rename (self):
    self.df = self.df.rename(columns={'dateApproval': 'date', 'corporateActionPrice': 'value'})


  def divide (self):
    self.df['value'] = 1 / self.df['value']

  def reset_index(self):
    self.df.reset_index(inplace=True)
    self.df.drop('index', axis=1, inplace=True)
    #print(self.df)



  def save_csv(self):
    self.df.to_csv(self.name +'.csv', index=False)



def main(df, name):
  df = Dataframe(df, name)
  df.slice()
  df.drop()
  df.transform_date()
  df.sort()
  df.replace()
  df.rename()
  df.divide()
  df.reset_index()
  df.drop()
  df.save_csv()





if __name__ == '__main__':
    #df_dividendo, df_capital, df_total = slice_df()
    main(df_total, 'df_total')
    main(df_capital, 'df_capital')
    main(df_dividendo, 'df_dividendo')


# COMMAND ----------

# MAGIC %md
# MAGIC ###Etapa de Machine Learning e Criação de Gráficos 

# COMMAND ----------

# MAGIC %%R
# MAGIC
# MAGIC devtools::install_github("itamarcaspi/psymonitor", force = FALSE)
# MAGIC
# MAGIC
# MAGIC install.packages("lubridate", repos = "http://cran.rstudio.com")
# MAGIC install.packages("gridExtra", repos = "http://cran.rstudio.com")
# MAGIC

# COMMAND ----------

# MAGIC %%R
# MAGIC library(psymonitor)
# MAGIC data(spread)

# COMMAND ----------

# MAGIC %%R
# MAGIC # trecho de código adaptado de Phillips & Shi(2020)
# MAGIC calculate <- function(spread, name){
# MAGIC
# MAGIC     library(psymonitor)
# MAGIC     library(gridExtra)
# MAGIC     library(grid)
# MAGIC     library(gtable)
# MAGIC     data(spread)
# MAGIC
# MAGIC     #spread  <- spread[1:150, ]
# MAGIC
# MAGIC
# MAGIC     y           <- spread$value #1/div
# MAGIC     spread$date <- as.Date(spread$date)
# MAGIC     obs         <- length(y)
# MAGIC     swindow0    <- floor(obs * (0.01 + 1.8 / sqrt(obs))) # set minimal window size
# MAGIC     IC          <- 2  # use BIC to select the number of lags
# MAGIC     adflag      <- 6  # set the maximum nuber of lags to 6
# MAGIC     yr          <- 2
# MAGIC     Tb          <- 12*yr + swindow0 - 1  # Set the control sample size
# MAGIC     nboot       <- 99  # set the number of replications for the bootstrap
# MAGIC
# MAGIC     bsadf <- PSY(y, swindow0 = swindow0, IC = IC, adflag = adflag)  # estimate the PSY test statistics sequence
# MAGIC     quantilesBsadf <- cvPSYwmboot(y, swindow0 = swindow0, IC = IC, adflag = adflag, Tb = Tb, nboot = 2, nCores = 2) # simulate critical values via wild bootstrap. Note that the number of cores is arbitrarily set to 2.
# MAGIC
# MAGIC     dim         <- obs - swindow0 + 1
# MAGIC     date        <- spread$date[swindow0:obs]
# MAGIC     quantile95  <- quantilesBsadf %*% matrix(1, nrow = 1, ncol = dim)
# MAGIC     ind95       <- (bsadf > t(quantile95[2, ])) * 1
# MAGIC     periods     <- locate(ind95, date)  # Locate crisis periods
# MAGIC     crisisDates <- disp(periods, obs)  #generate table that holds crisis periods
# MAGIC
# MAGIC
# MAGIC     titulo <- textGrob(sprintf("Episódios de Crises das Empresas Listadas na B3:\n %s", name), gp = gpar(fontsize = 15, fontface = "bold"))
# MAGIC     tamanho_topo <- unit(2, "in")  # Neste exemplo, usamos 2 polegadas
# MAGIC
# MAGIC
# MAGIC     tabela      <- tableGrob(crisisDates, theme = ttheme_default(base_size = 40))  
# MAGIC     grid.arrange(titulo, tabela, top = tamanho_topo)
# MAGIC     return(ind95)
# MAGIC
# MAGIC }

# COMMAND ----------

# MAGIC %%R
# MAGIC # trecho de código adaptado de Phillips & Shi(2020)
# MAGIC chart <- function(ind95, spread, name) {
# MAGIC
# MAGIC    
# MAGIC
# MAGIC     y        <- spread$value #1/div
# MAGIC     spread$date     <- as.Date(spread$date)
# MAGIC     obs      <- length(y)
# MAGIC     swindow0 <- floor(obs * (0.01 + 1.8 / sqrt(obs))) 
# MAGIC     date         <- spread$date[swindow0:obs]
# MAGIC
# MAGIC     plot(date,y[swindow0:obs],xlim=c(min(date),max(date)), main = sprintf("Episódios de Crises das Empresas Listadas na B3:\n %s", name), ylim=c(0.1,8), type='l',lwd=3, xlab = "", ylab = "")
# MAGIC     for(i in 1:length(date)){
# MAGIC         if (ind95[i]==1){abline(v=date[i],col=3)}
# MAGIC     }
# MAGIC     points(date,y[swindow0:obs],type='l')
# MAGIC     box(lty=1)
# MAGIC }

# COMMAND ----------

# MAGIC %%R
# MAGIC obs_data <- function(spread, name) {
# MAGIC     library(lubridate)
# MAGIC     library(gridExtra)
# MAGIC
# MAGIC     spread$date <- as.Date(spread$date)
# MAGIC     tabela_frequencia <- table(format(spread$date, "%Y"))
# MAGIC     df_tabela <- as.data.frame(tabela_frequencia)
# MAGIC
# MAGIC     colnames(df_tabela) <- c("Ano", "Contagem") 
# MAGIC
# MAGIC     titulo <- textGrob(sprintf("Contagem de Observações por Ano na B3:\n %s", name), gp = gpar(fontsize = 15, fontface = "bold"))
# MAGIC
# MAGIC
# MAGIC     n <- nrow(df_tabela)
# MAGIC     n_colunas <- 2
# MAGIC     n_linhas <- ceiling(n / n_colunas)
# MAGIC     metade1 <- df_tabela[1:n_linhas, ]
# MAGIC     metade2 <- df_tabela[(n_linhas + 1):n, ]
# MAGIC
# MAGIC     tabela1 <- tableGrob(metade1, theme = ttheme_default(base_size = 17))  
# MAGIC     tabela2 <- tableGrob(metade2, theme = ttheme_default(base_size = 17))  
# MAGIC
# MAGIC     tamanho_topo <- unit(2, "in")  
# MAGIC
# MAGIC     
# MAGIC     grid.arrange(titulo,
# MAGIC                  arrangeGrob(tabela1, tabela2, ncol = n_colunas),
# MAGIC                  top = tamanho_topo,
# MAGIC                  heights = c(0.1, 0.9))
# MAGIC }
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %%R
# MAGIC spread <- read.csv("df_dividendo.csv")
# MAGIC name <- "Dividend Yield"
# MAGIC ind95 <- calculate(spread, name)

# COMMAND ----------

# MAGIC %%R
# MAGIC chart(ind95, spread, name="Dividend Yield")

# COMMAND ----------

# MAGIC %%R
# MAGIC spread <- read.csv("df_dividendo.csv")
# MAGIC obs_data(spread, name)

# COMMAND ----------

# MAGIC %%R
# MAGIC spread <- read.csv("df_capital.csv")
# MAGIC name <- "Dividend JSCP yield"
# MAGIC ind95 <- calculate(spread, name)
# MAGIC

# COMMAND ----------

# MAGIC %%R
# MAGIC name <- "Dividend JSCP yield"
# MAGIC chart(ind95, spread, name)
# MAGIC

# COMMAND ----------

# MAGIC %%R
# MAGIC spread <- read.csv("df_capital.csv")
# MAGIC obs_data(spread, name)

# COMMAND ----------

# MAGIC %%R
# MAGIC spread <- read.csv("df_total.csv")
# MAGIC name <- "Total = Dividend + JSCP"
# MAGIC ind95 <- calculate(spread, name)

# COMMAND ----------

# MAGIC %%R
# MAGIC name <- "Total = Dividendos + JSCP"
# MAGIC chart(ind95, spread, name)
# MAGIC

# COMMAND ----------

# MAGIC %%R
# MAGIC spread <- read.csv("df_total.csv")
# MAGIC obs_data(spread, name)

# COMMAND ----------


