 2/1: import pandas as pd
 3/1: ls
 3/2: cd assets/
 3/3: ls
 3/4: import pandas as pd
 3/5: df = pd.read_csv('./datasets_7486_10652_featuresdf (1).csv')
 3/6: df.head(2)
 3/7: df.head(1)
 3/8: df.columns
 3/9: df.columns
3/10: df.describe()
3/11: df.describe(include = "all")
3/12: ls
3/13: df.head(3)
3/14: df.info()
3/15: pip
3/16: pip install numpy
3/17: import numpy as np
3/18: pip install matplotlib
3/19: ls
3/20: import matplotlib.pyplot as plt
3/21: pip install seaborn
3/22: ls
3/23: import seaborn as sns
3/24: f, aux = plt.subplots(figsize=(18,18))
3/25: sns.heatmap(df.corr(), annot=True, linewidths = .5, fmt = '.1f', ax=aux)
3/26: ls
3/27: ls
3/28: ls
3/29: cd ls
3/30: cd
3/31: ls
3/32: ls
3/33: df
 5/1: import pandas as pd
 5/2: df  = pd.read_csv('./assets/datasets_7486_10652_featuresdf (1).csv')
 5/3: df.head(4)
 5/4: df.columns()
 5/5: df['name'].describe()
 5/6: df['name'].describe(3)
 5/7: df['name'].describe()
 5/8: df['name'].describe() # música mais frequente
 5/9: df['artist'].describe() # artista mais frequente
5/10: df['artists'].describe() # artista mais frequente
5/11:
df.drop(columns=['id']) # removendo coluna desnecessária
df.head()
5/12:
df.drop(columns=['id'], inplace=True) #removendo coluna desnecessária
df.head()
 6/1:
#Linear Models Example

from sklearn import linear_model
 6/2:
#Linear Models Example

from sklearn import linear_model
 6/3:
#Linear Models Example

from sklearn import linear_model

x = [[0., 0.], [1.,1.], [2.,2.],[3.,3.]]
y = [0.,1.,2.,3.]
reg = linear_model.BayesianRidge()
reg.fit(x,y)
BayesianRidge()
 6/4:
#Linear Models Example

from sklearn import linear_model

x = [[0., 0.], [1.,1.], [2.,2.],[3.,3.]]
y = [0.,1.,2.,3.]
reg = linear_model.BayesianRidge()
reg.fit(x,y)
 6/5: reg.predict([[1,0.]])
 6/6: reg.coef_
 6/7:
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import BayesianRidge, LinearRegression

# #############################################################################
# Generating simulated data with Gaussian weights
np.random.seed(0)
n_samples, n_features = 100, 100
X = np.random.randn(n_samples, n_features)  # Create Gaussian data
# Create weights with a precision lambda_ of 4.
lambda_ = 4.
w = np.zeros(n_features)
# Only keep 10 weights of interest
relevant_features = np.random.randint(0, n_features, 10)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
# Create noise with a precision alpha of 50.
alpha_ = 50.
noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
# Create the target
y = np.dot(X, w) + noise

# #############################################################################
# Fit the Bayesian Ridge Regression and an OLS for comparison
clf = BayesianRidge(compute_score=True)
clf.fit(X, y)

ols = LinearRegression()
ols.fit(X, y)

# #############################################################################
# Plot true weights, estimated weights, histogram of the weights, and
# predictions with standard deviations
lw = 2
plt.figure(figsize=(6, 5))
plt.title("Weights of the model")
plt.plot(clf.coef_, color='lightgreen', linewidth=lw,
         label="Bayesian Ridge estimate")
plt.plot(w, color='gold', linewidth=lw, label="Ground truth")
plt.plot(ols.coef_, color='navy', linestyle='--', label="OLS estimate")
plt.xlabel("Features")
plt.ylabel("Values of the weights")
plt.legend(loc="best", prop=dict(size=12))

plt.figure(figsize=(6, 5))
plt.title("Histogram of the weights")
plt.hist(clf.coef_, bins=n_features, color='gold', log=True,
         edgecolor='black')
plt.scatter(clf.coef_[relevant_features], np.full(len(relevant_features), 5.),
            color='navy', label="Relevant features")
plt.ylabel("Features")
plt.xlabel("Values of the weights")
plt.legend(loc="upper left")

plt.figure(figsize=(6, 5))
plt.title("Marginal log-likelihood")
plt.plot(clf.scores_, color='navy', linewidth=lw)
plt.ylabel("Score")
plt.xlabel("Iterations")


# Plotting some predictions for polynomial regression
def f(x, noise_amount):
    y = np.sqrt(x) * np.sin(x)
    noise = np.random.normal(0, 1, len(x))
    return y + noise_amount * noise


degree = 10
X = np.linspace(0, 10, 100)
y = f(X, noise_amount=0.1)
clf_poly = BayesianRidge()
clf_poly.fit(np.vander(X, degree), y)

X_plot = np.linspace(0, 11, 25)
y_plot = f(X_plot, noise_amount=0)
y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)
plt.figure(figsize=(6, 5))
plt.errorbar(X_plot, y_mean, y_std, color='navy',
             label="Polynomial Bayesian Ridge Regression", linewidth=lw)
plt.plot(X_plot, y_plot, color='gold', linewidth=lw,
         label="Ground Truth")
plt.ylabel("Output y")
plt.xlabel("Feature X")
plt.legend(loc="lower left")
plt.show()
 7/1: import matplotlib.pyplot as plt
 7/2:
x = df['loudness']
y = df['duration_ms']
 7/3: import pandas as pd
 7/4: df  = pd.read_csv('./assets/datasets_7486_10652_featuresdf (1).csv')
 7/5: df  = pd.read_csv('../assets/datasets_7486_10652_featuresdf (1).csv')
 7/6: df['name'].describe() # música mais frequente
 7/7: df['artists'].describe() # artista mais frequente
 7/8:
df.drop(columns=['id'], inplace=True) #removendo coluna desnecessária
df.head()
 7/9:
x = df['loudness']
y = df['duration_ms']
7/10: import matplotlib.pyplot as plt
7/11: plt.scatter(x,y)
7/12:
x = df['instrumentalness']
y = df['duration_ms']
7/13: plt.scatter(x,y)
7/14:
plt.title('Correlação entre instrumentalidade e duração')
plt.xlabel('instrumentalidade')
plt.ylabel('duração')
7/15:
plt.title('Correlação entre instrumentalidade e duração')
plt.xlabel('instrumentalidade')
plt.ylabel('duração')
plt.scatter(x,y)
10/1: import pandas as pd
10/2: df  = pd.read_excel('./HIST_PAINEL_COVIDBR_15jul2020')
10/3: l
10/4: la
10/5: ls
11/1: import pandas as pd
11/2: df = pd.read_csv('./HIST_PAINEL_COVIDBR_15jul2020 (2).csv')
11/3: df = pd.read_csv('./HIST_PAINEL_COVIDBR_15jul2020 (2) - HIST_PAINEL_COVIDBR_15jul2020 (2).csv')
11/4: df.head()
11/5: df['municipio']
11/6: df.head()
11/7: df['regiao']
11/8: ls
11/9: df = pd.read_excel(r './HIST_PAINEL_COVIDBR_15jul2020.xlsx')
11/10: df = pd.read_excel(r"./HIST_PAINEL_COVIDBR_15jul2020.xlsx")
11/11: df = pd.read_excel("./HIST_PAINEL_COVIDBR_15jul2020.xlsx")
11/12: pip install xlrd
11/13: df = pd.read_excel("./HIST_PAINEL_COVIDBR_15jul2020.xlsx")
11/14: df.head()
11/15: df['regiao']
11/16: df['regiao'].describe()
11/17: df
11/18: df['regiao'].query('Montes')
11/19: df.loc['Minas']
11/20: df['regiao']
11/21: regiao = df['regiao']
11/22: type(regiao)
11/23: regiao
11/24: regiao[10]
11/25: regiao[100]
11/26: regiao[400]
11/27: regiao.count()
11/28: set(regiao)
11/29: regiao
11/30: df.head()
11/31: df['estado']
11/32: df['estado']['MG']
11/33: df['estado'].corr()
11/34: ls
12/1: import instaloader
12/2: USER = 'kaiofprates'
12/3: PROFILE = USER
12/4: L = instaloader.Instaloader()
12/5: L.load_session_from_file(USER)
13/1: username = "dorksfacomp"
13/2: password = "alteredCarbon14"
13/3: import instaloader
13/4: L = instaloader.Instaloader()
13/5: L.login(username,password)
13/6: profile = instaloader.Profile.get_followees(L)
13/7: profile
13/8: profile = instaloader.Profile.from_username(L.context)
13/9: profile = instaloader.Profile.from_username(L.context, 'dorksfacomp')
13/10: profile.get_followers()
13/11: set(profile.get_followers())
13/12: followers = set(profile.get_followers())
13/13: followers
13/14: type(followers)
14/1: username = "kaiofprates"
14/2: password = "kikoiq89"
14/3: import instaloader
14/4: L = instaloader.Instaloader()
14/5: L.login(username, password)
14/6: profile = instaloader.Profile.from_username(L.context, username)
14/7: followers = set(profile.get_followers())
14/8: len(followers)
15/1: import ins
17/1: username = "kaiofprates"
17/2: password = "kikoiq89"
17/3: import instaloader
17/4: L = instaloader.Instaloader()
17/5: L.login(username, password)
17/6: profile = instaloader.Profile.from_username(L.context, username)
17/7: followers = set(profile.get_followers())
17/8: type(followers)
17/9:
for i in followers: 
    print i
17/10:
for i in followers: 
    print(i)
17/11: followers[0]
17/12: followers = list(followers)
17/13: len(followers)
17/14:
followers[
0]
17/15: followers[0]
17/16: followers[0].split(' ')
17/17: str(followers[0]).split(' ')[2]
17/18: str(followers[0]).split(' ')[1]
18/1: ls
18/2: cd app
18/3: ls
18/4: import Insta from Api
18/5: import Api
18/6: import Api
18/7: insta = Api('dorksfacomp', 'alteredCarbon14')
18/8: insta = new Api('dorksfacomp', 'alteredCarbon14')
18/9: from Api import Insta
18/10: insta = Insta('dorksfacomp', 'alteredCarbon14')
18/11: insta.login()
18/12: insta = Insta('dorksfacomp', 'alteredCarbon14')
18/13: insta.login()
18/14: insta.login
19/1: from Api import Insta
19/2: ls
19/3: cd app/
19/4: ls
19/5: from Api import Insta
19/6: insta = Insta('dorksfacomp', 'alteredCarbon14')
19/7: insta.login()
19/8: from Api import Insta
19/9: insta = Insta('dorksfacomp', 'alteredCarbon14')
19/10: insta.login()
19/11: from Api import Insta
19/12: insta = Insta('dorksfacomp', 'alteredCarbon14')
19/13: insta.login()
19/14: from Api import Insta
19/15: insta = Insta('dorksfacomp', 'alteredCarbon14')
19/16: insta.login()
20/1: cd A
20/2: ls
20/3: cd app/
20/4: ls
20/5: from Api import Insta
20/6: insta = Insta('dorksfacomp', 'alteredCarbon14')
20/7: insta.login()
21/1: cd app/
21/2: ls
21/3: from Api import Insta
21/4: from Api import Insta
21/5: insta = Insta('dorksfacomp', 'alteredCarbon14')
21/6: insta.login()
21/7: from Api import Insta
21/8: insta = Insta('dorksfacomp', 'alteredCarbon14')
21/9: insta.login()
22/1: cd app/
22/2: ls
22/3: from Api import Insta
22/4: insta = Insta('dorksfacomp', 'alteredCarbon14')
22/5: insta.login()
22/6: insta.get_followeers()
23/1: from Api import Insta
23/2: insta = Insta('dorksfacomp', 'alteredCarbon14')
23/3: insta.login()
23/4: insta.get_followeers()
24/1: from Api import Insta
24/2: insta = Insta('dorksfacomp', 'alteredCarbon14')
24/3: insta.login()
24/4: insta.get_followeers()
24/5: insta.get_info()
25/1: import pandas as pd
25/2: df = pd.read_excel('./HIST_PAINEL_COVIDBR_15jul2020.xlsx')
25/3: import matplotlib.pylab as plt
25/4: df.head()
25/5: df.head(10)
25/6: missing_data = df.isnull()
25/7: missing_data.head()
25/8:
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
25/9: avg_norm_loss = df['emAcompanhamentoNovos'].astype('int64').mean(axis=0)
25/10: avg_norm_loss = df['emAcompanhamentoNovos'].astype('float').mean(axis=0)
25/11: avg_norm_loss
25/12: import numpy as np
25/13: df['emAcompanhamentoNovos'].replace(np.nan, avg_norm_loss, inplace=True)
25/14: df.head(10)
25/15: df.types()
25/16: df.values()
25/17: df.dtypes
25/18: df['populacaoTCU2019'].head(4)
25/19: df['populacaoTCU2019'].astype('float')
25/20: df['populacaoTCU2019'].astype( int, copy = True)
25/21: avg_populacaoTCU2019 = df ['populacaoTCU2019'].astype().mean(axis=0)
25/22: avg_populacaoTCU2019 = df ['populacaoTCU2019'].astype('float').mean(axis=0)
25/23: avg_populacaoTCU2019 = df ['populacaoTCU2019'].astype('int').mean(axis=0)
25/24: df['populacaoTCU2019'].replace(np.nan, 0.0 , inplace=True)
25/25: df['populacaoTCU2019'].head(10)
25/26: avg_populacaoTCU2019 = df ['populacaoTCU2019'].astype('float').mean(axis=0)
25/27: df['populacaoTCU2019'].astype( int, copy = True)
25/28: df['populacaoTCU2019'].astype( float, copy = True)
25/29: avg_populacaoTCU2019 = df ['populacaoTCU2019'].astype().mean(axis=0)
25/30: plt
25/31: df.columns
25/32: plt.pyplot.hist(df['casosNovos'])
25/33: from matplotlib import pyplot
25/34: plt.pyplot.hist(df['casosNovos'])
25/35: import matplotlib as plt
25/36: from matplotlib impor pyplot
25/37: from matplotlib import pyplot
25/38: plt.pyplot.hist(df['casosNovos'])
25/39: plt.pyplot
25/40: plt.pyplot.xlabel('casos novos')
25/41: plt.pyplot.ylabel('contagem')
25/42: plt.pyplot.title('Contagem de novos casos de COVID-19')
25/43: plt
25/44: plt.pyplot()
25/45: plt.show()
25/46: plt.pyplot.show()
25/47: df.head()
25/48: df.head(10)
25/49: import matplotlib.pyplot as plt
25/50: plt.plot(df['casosNovos'])
25/51: plt.ylabel('progression of cases')
25/52: plt.show()
25/53: df.columns
25/54: plt.axis(df['data'])
25/55: import matplotlib.pyplot as plt
25/56: df.columns
25/57: x = df['casosNovos']
25/58: y = df['data']
25/59: plt.scatter(x,y)
25/60: plt.title('Relação de casos por data')
25/61: plt.xlabel('casos novos')
25/62: plt.ylabel('data')
25/63: plt.show()
25/64: plt.scatter(y,x)
25/65: plt.show()
25/66: plt.xlabel('data')
25/67: plt.ylabel('casos novos')
25/68: plt.show()
25/69: plt.scatter(y,x)
25/70: plt.xlabel('data')
25/71: plt.ylabel('casos novos')
25/72: plt.show()
26/1: import requests
26/2: profile = requests.get('https://www.instagram.com/kaiofprates/')
26/3: profile
26/4: profile.text()
26/5: profile.text
26/6: profile.content
26/7: pip install beautifulsoup
26/8: pip install beatutifulsoup4
26/9: pip install beautifulsoup4
26/10: from bs4 import BeautifulSoup
26/11: soup = BeautifulSoup(profile.content)
26/12: tag = soup.b
26/13: type(tag)
26/14: soup = BeautifulSoup(profile.text)
26/15: tag = soup.b
26/16: type(tag)
26/17: tag
26/18: soup
26/19: soup.b
26/20: soup.getText()
26/21: soup.p['class']
26/22: soup['class']
26/23: soup.prettify()
26/24: soup = BeautifulSoup(profile.content, 'html.parser')
26/25: soup.prettify()
26/26: soup = BeautifulSoup(profile.content, 'html.parser')
26/27: soup.b
26/28: soup.p
26/29: soup.title
26/30: soup.title.name
26/31: soup.title.string
26/32: soup.find_all('a')
26/33: soup.find_all('class')
26/34: soup.find_all('div')
26/35: soup.find_all('zwlfE')
26/36: soup.find_all('class')
26/37: soup.find_all('div')
26/38: soup.find_all('div')[2]
26/39: soup.find_all('div')[0]
26/40: type(soup.find_all('div')[0])
26/41: soup.find_all('div')[0]
26/42: soup.find_all('div')[0][1]
26/43: soup.find_all('div')[0].span
26/44: soup.find('class')
26/45: soup
26/46: soup.link
26/47: soup.class
26/48: soup.div
26/49: soup.div.div
26/50: soup.div.span
26/51: pip install urllib
26/52: soup = BeautifulSoup(profile.text)
26/53: profile
26/54: profile.url
26/55:
for script in soup(['script', 'stype']):
    script.extract()
26/56: text = soup.get_text()
26/57: lines = ( line.strip() from line in text.splitlines())
26/58: lines = ( line.strip() for line in text.splitlines())
26/59: chunks = ( phrase.strip() for line in lines for phrase in line.split(" "))
27/1: pip install nltk
27/2: import nltk
27/3: import requests
27/4: profile = requests.get('https://www.instagram.com/kaiofprates/')
27/5: text = nltk.clean_html(profile.text)
27/6: from bs4 import BeautifulSoup
27/7: soup = BeautifulSoup(profile.text)
27/8: text = nltk.clean_html(soup.get_text())
27/9: soup = BeautifulSoup(profile.content)
27/10: text = nltk.clean_html(soup.get_text())
27/11: pip install urllib
27/12: pip install urllib3
27/13: cd ..
28/1: import  requests
28/2: from bs4 import BeautifulSoup
28/3: URL = "https://www.instagram.com/kaiofprates/"
28/4: URL
28/5: r = requests.get(URL)
28/6: s = BeautifulSoup(r.text, "html.parser")
28/7: meta = s.find('meta', property="og:description")
28/8: meta.attrs['content']
28/9: s
29/1: import instaloader
29/2: user = 'dorksfacomp'
29/3: pass = 'alteredCarbon14'
29/4: pw = 'alteredCarbon14'
29/5: L = instaloader.Instaloader()
29/6: L.login(user, pw)
29/7: profile = instaloader.Profile.from_username(L.context, user)
29/8: profile.get_followees()
29/9: followes = set( profile.get_followees())
29/10: followes
30/1: import instaloader
30/2: pw = 'alteredCarbon14'
30/3: user = 'dorksfacomp'
30/4: L = instaloader.Instaloader()
30/5: L.login(user, pw)
30/6: profile = instaloader.Profile.from_username(L.context, user)
30/7: profile.biography
30/8: profile.get_profile_pic_url()
30/9: profile.get_profile_pic_url()
31/1: import instaloader
31/2: user = 'dorksfacomp'
31/3: pw = 'alteredCarbon14'
31/4: L.login(user, pw)
31/5: L = instaloader.Instaloader()
31/6: L.login(user, pw)
31/7: profile = instaloader.Profile.from_username(L.context, user)
31/8: followes = set( profile.get_followees())
31/9: followes
31/10:
for i in followes: 
    print(i.get_profile_pic_url())
31/11: profile.full_name
32/1: import time
32/2: import pyautogui
32/3:
def clicar():
    time.sleep(3)
    pyautogui.write('eita porra')
32/4: clicar()
33/1:
followees =  [
      "ethicalhackingpro",
      "ethical_hacking_trickinfo",
      "kaiofprates",
      "luizpaulo.rodrigues",
      "pabloferraz1998",
      "hacker_union",
      "ethical__hacking",
      "learn_ethical_hacking_n_other_"
    ]
33/2: followees
33/3:
followers = 
      "nilo_cintia",
      "lana_construtora",
      "muhil12345",
      "laressadealkmim",
      "prasad_vagmare",
      "lorranealkimim",
      "maniapromocoes",
      "drillhacks",
      "pabloferraz1998",
      "kaiofprates",
      "deltree_hackers",
      "danpudndcvqmmblou76z3ssrt",
      "cerqueiracarloscezarguedes",
      "naldoreiteixeira",
      "hzfahysbd_",
      "vinicao_gomes",
      "nadson.rocha.12382",
      "nslover546"
    ]
33/5:
followers = 
      "nilo_cintia",
      "lana_construtora",
      "muhil12345",
      "laressadealkmim",
      "prasad_vagmare",
      "lorranealkimim",
      "maniapromocoes",
      "drillhacks",
      "pabloferraz1998",
      "kaiofprates",
      "deltree_hackers",
      "danpudndcvqmmblou76z3ssrt",
      "cerqueiracarloscezarguedes",
      "naldoreiteixeira",
      "hzfahysbd_",
      "vinicao_gomes",
      "nadson.rocha.12382",
      "nslover546"]
33/6:
followers = [
      "nilo_cintia",
      "lana_construtora",
      "muhil12345",
      "laressadealkmim",
      "prasad_vagmare",
      "lorranealkimim",
      "maniapromocoes",
      "drillhacks",
      "pabloferraz1998",
      "kaiofprates",
      "deltree_hackers",
      "danpudndcvqmmblou76z3ssrt",
      "cerqueiracarloscezarguedes",
      "naldoreiteixeira",
      "hzfahysbd_",
      "vinicao_gomes",
      "nadson.rocha.12382",
      "nslover546"]
33/7: set(followees).intersection(followers)
33/8: set(followers).intersection(followees)
33/9: set(followees).difference(followers)
35/1: import pandas as pd
35/2: import numpty as np
35/3: import numpy as np
35/4: import matplotlib.pyplot as plt
35/5: path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
35/6: df = pd.read_csv(path)
35/7: df
35/8: df.head()
35/9: from sklearn.linear_model import LinearRegression
35/10: lm = LinearRegression()
35/11: lm
35/12: X = def[['highway-mpg']]
35/13: X = df[['highway-mpg']]
35/14: Y = df['price']
35/15: lm.fit(X,Y)
35/16: Yhat = lm.predict(X)
35/17: Yhat
35/18: Yhat[0:5]
35/19: lm.intercept_
35/20: lm.coef_
35/21: price = 38423.31 - 821.73 * 0.31
35/22: price
35/23: df['engine-size']
35/24: df[['engine-size']]
35/25: cd /home/kaiofprates/Developer
35/26: Z  = df[['horsepower' , 'curb-weight' , 'engine-size' , 'highway-mpg']]
35/27: lm.fit(Z, df['price'])
35/28: lm.coef_
35/29: lm.intercept_
35/30: import seaborn as sns
35/31: width = 12
35/32: height  = 10
35/33: plt.figure(figsize=(width,height))
35/34: sns.regplot(x = 'highway-mpg' , y = 'price', data=df)
35/35: plt.ylim(0,)
35/36: plt.show()
35/37: ls
35/38: plt.figure(figsize=(width,height) )
35/39: sns.regplot(x='peak-rpm',y='price', data=df)
35/40: plt.ylim(0,)
35/41: plt.show()
35/42: df[['peak-rpm', 'highway-mpg', 'price' ]].corr()
36/1: ls
36/2: cd Spotify/
36/3: ls
36/4: ls
36/5: time
36/6: import pandas as pd
36/7: df = pd.read_csv('datasets_293841_602591_top50.csv')
36/8: df = pd.read_csv('datasets_293841_602591_top50.csv', encoding = "ISO-8859-1")
36/9: df
36/10: df.columns
36/11: df['Energy', 'Popularity','Genere'].corr()
36/12: df[['Energy', 'Popularity','Genere']].corr()
36/13: df.columns
36/14: df[['Energy', 'Popularity','Genre']].corr()
36/15: df[['Energy', 'Popularity'], 'Genre'].corr()
36/16: ls
36/17: df[['Energy', 'Popularity','Genre']].corr()
36/18: df['Genre']
36/19: df['Genre'].count()
36/20: df['Genre'].describec()
36/21: df['Genre'].describe()
36/22: df['Genre'].dtypes()
36/23: df['Genre'].dtypes
36/24: df['Genre'].dtype()
36/25: df['Genre'].dtype
36/26: df['Genre'].duplicated()
36/27: df['Genre']
36/28: generos  = set(df['Genre'])
36/29: generos
36/30: len(generos)
36/31: ls
36/32: import matplotlib.pylab as plt
36/33: ls
36/34: import numpy as np
36/35: df
36/36: missing_data = df.isnull()
36/37: missing_data
36/38:
for column in missing_data.columns.value.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print(" ")
36/39:
for column in missing_data.columns.value.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print(" ")
36/40: df.header()
36/41: df.head()
36/42: df['Popularity']
36/43: df['Popularity']
36/44: df['Popularity']
36/45: bins  = np.linspace(min(df['Popularity']), max(df['Popularity']),4)
36/46: bins
36/47: group_names = ['Low','Medium','High']
36/48: df['Popylarity-binned'] = pd.cut(df['Popularity'] , bins, labels=group_names, include_lowest=True )
36/49: df[['Popularity','Popylarity-binned']].head()
36/50: df
36/51: df.rename( index = { 'Popylarity-binned' : 'Popularity-binned'} , inplace=True)
36/52: df.head()
36/53: df.rename( columns = { 'Popylarity-binned' : 'Popularity-binned'} , inplace=True)
36/54: df.head()
36/55: from matplotlib import pyplot
36/56: plt.pyplot.hist(df['Popularity'])
36/57: import matplotlib from plt
36/58: import matplotlib as plt
36/59: import matplotlib as plt
36/60: from matplotlib import pyplot
36/61: plt.pyplot.hist(df['Popularity'])
36/62: plt.pyplot.xlabel('popularity')
36/63: plt.pyplot.ylabel('count')
36/64: plt.pyplot.title('Index of Popularity of top 50 musics')
36/65: plt.show()
36/66: plt.show
36/67: plt.pyplot.show()
36/68: df
36/69: df.to_csv('data_binning.csv')
36/70: ls
36/71: git
37/1: import pandas as pd
37/2: import numpy as np
37/3: path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
37/4: df = pd.read_csv(path)
37/5: df.head()
37/6: import matplotlib.pyplot as plt
37/7: import seaborn as sns
37/8: print(df.dtypes)
37/9: df.corr()
37/10: df[['bore','stroke','compression-ratio','horsepower']].corr()
37/11: sns.regplot(x="engine-size",y="price", data=df)
37/12: plt.ylim(0,)
37/13: plt.show()
37/14: df[['engine-size','price']].corr()
37/15: sns.regplot(x="highway-mpg", y="price", data=df)
37/16: plt.show()
37/17: df[['highway-mpg','price']].corr()
37/18: sns.regplot(x="peak-rpm" , y="price" ,data=df)
37/19: plt.show()
37/20: df[['peak-rpm' , 'price']].corr()
37/21: df[['stroke','price']].corr()
37/22: sns.regplot(x='stroke',y='price', data=df)
37/23: plt.show()
37/24: sns.boxplot(x='body-style', y = 'price' , data=df)
37/25: plt.show()
37/26: sns.boxplot(x='engine-location', y='price', data=df)
37/27: plt.show()
37/28: sns.boxplot(x="drive-wheels",y="price",data=df)
37/29: plt.show()
37/30: df.describe()
37/31: df.describe(include=['object'])
37/32: df['drive-wheels'].value_counts()
37/33: drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
37/34: drive_wheels_counts.rename(columns={ 'drive-wheels' : 'value_counts' }, inplace=True)
37/35: drive_wheels_counts
37/36: drive_wheels_counts.index.name = "drive-wheels"
37/37: drive_wheels_counts
37/38: engine_loc_counts = df['engine-location'].value_counts().to_frame()
37/39: engine_loc_counts.rename(columns={'engine-location' : 'value_counts'} , inplace = True)
37/40: engine_loc_counts.index.name = 'engine-location'
37/41: engine_loc_counts.head(10)
37/42: df['drive-wheels'].unique()
37/43: df_group_one = df[['drive-wheel', 'body-style' ,'price']]
37/44: df_group_one = df[['drive-wheels', 'body-style' ,'price']]
37/45: df_group_one = df_group_one.groupby(['drive-wheels'] , as_index=False).mean()
37/46: df_group_one
37/47: df_gptest = df[['drive-wheels','body-style' ,' price']]
37/48: df_gptest = df[['drive-wheels','body-style' ,' price']]
37/49: df_gptest = df[['drive-wheels','body-style' ,'price']]
37/50: grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
37/51: grouped_test1
37/52: grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')
37/53: grouped_pivot
37/54: grouped_pivot = grouped_pivot.fillna(0)
37/55: grouped_pivot
37/56: df_pgtest2 = df[['body-style', 'price']]
37/57: grouped_test2 = df_pgtest2.groupby(['body-style'], as_index=False).mean()
37/58: grouped_test2
37/59: import matplotlib.pyplot as plt
37/60: plt.pcolor(grouped_pivot, cmap='RdBu')
37/61: plt.colorbar()
37/62: plt.show()
37/63: fig, as = plt.subplots()
37/64: fig, ax = plt.subplots()
37/65: im = ax.pcolor(grouped_pivot, cmap='RdBu')
37/66: row_labels = grouped_pivot.columns.levels[1]
37/67: col_labels = grouped_pivot.index
37/68:
ax.set_xticks(np.arange(grouped_pivot.shape[1] + 0.5, minor = False)
)
37/69: ax.set_xticks(np.arange(grouped_pivot.shape[1] )+ 0.5, minor = False)
37/70: ax.set_yticks(np.arange(grouped_pivot.shape[0] )+ 0.5, minor = False)
37/71: ax.set_xticklabels(row_labels, minor=False)
37/72: ax.set_yticklabels(col_labels, minor=False)
37/73: plt.xticks(rotation=90)
37/74: fig.colorbar(im)
37/75: plt.show()
37/76: from scipy import stats
37/77: pearson_coef, p_value  = stats.pearsonr(df['wheel-base'], df['price'])
37/78: print('The Pearson Correlation Coefficiente is', pearson_coef, " with a P-value of  P =", p_value)
37/79: test2=df_gptest[['drive-wheels' , 'price']].groupby(['drive-wheels'])
37/80: test2.head(2)
37/81: grouped_test2.get_group('4wd')['price']
38/1: import pandas as pd
38/2: import numpy as np
38/3: df = pd.read_csv('datasets_293841_602591_top50.csv')
38/4: df = pd.read_csv('data_binning.csv')
38/5: df
38/6: df.corr()
38/7: import matplotlib.pyplot as plt
38/8: import seaborn as sns
38/9: sns.regplot(x='Danceability', y='Popularity', data=df)
38/10: plt.ylim(0,)
38/11: plt.show()
38/12: df['Danceability'] = df['Danceability'] /df['Danceability'].max()
38/13: df['Popularity'] = df['Popularity'] / df['Popularity'].max()
38/14: sns.regplot(x='Danceability', y='Popularity', data=df)
38/15: plt.ylim(0,)
38/16: plt.show()
38/17: df.head()_
38/18: df.head()
38/19: sns.regplot(x='Genre', y='Popularity', data=df)
38/20: df.corr()
38/21: sns.regplot(x='Length', y='Popularity', data=df)
38/22: sns.regplot(x='Length', y='Popularity', data=df)
38/23: df.corr()
38/24: sns.regplot(x='Energy', y='Popularity', data=df)
38/25: plt.ylim(0,)
38/26: plt.show()
38/27: sns.regplot(x='Beats.Per.Minute', y = 'Danceability', data=df)
38/28: plt.ylim(0,)
38/29: plt.show()
40/1: import instaloader
40/2: l = instaloader.Instaloader()
40/3: profile = instaloader.Profile.from_username('tarajiphenson')
40/4: l.login('kaioprates_dev','césio137')
40/5: profile = instaloader.Profile.from_username( l.context, 'tarajiphenson')
40/6: profile.biography()
40/7: profile.biography
40/8: profile = instaloader.Profile.from_username( l.context, 'kaioprates_dev')
40/9: profile.
41/1: import pandas as pd
41/2: data  = pd.read_excel
41/3: data  = pd.read_excel('HIST_PAINEL_COVIDBR_10ago2020 (1).xlsx')
41/4: data
41/5: data.keys()
41/6: len(data.keys())
41/7: data.head(30)
41/8: data.tail(30)
41/9: obito = data['obitosNovos']
41/10: obito
41/11: sum(obito)
42/1: import pandas as pd
42/2: mega  = pd.read_excel('mega_sena_asloterias_ate_concurso_2289_sorteio.xlsx')
42/3: mega.head()
42/4: mega.head(10)
42/5: mega.keys()
42/6: mega['Unnamed: 1']
42/7: mega['Unnamed: 2']
42/8: mega[['Unnamed: 2', 'Unnamed: 3']]
42/9: mega[['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5','Unnamed: 6','Unnamed: 7']]
42/10: n = mega[['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5','Unnamed: 6','Unnamed: 7']]
42/11: set(n)_
42/12: set(n)
42/13: n
42/14: n['Unnamed: 2']
42/15: n['Unnamed: 2'].bool
42/16: n['Unnamed: 2']
42/17: set(n['Unnamed: 2'])
42/18: n = list(n['Unnamed: 2'])
42/19: n
42/20: n = mega[['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5','Unnamed: 6','Unnamed: 7']]
42/21: n1 = list(n['Unnamed: 2'])
42/22: n1 = list(n['Unnamed: 3'])
42/23: n1 = list(n['Unnamed: 2'])
42/24: n2 = list(n['Unnamed: 3'])
42/25: n3 = list(n['Unnamed: 4'])
42/26: n4 = list(n['Unnamed: 5'])
42/27: n5 = list(n['Unnamed: 6'])
42/28: n6 = list(n['Unnamed: 7'])
42/29: n1  + n2
42/30: [1,2,3] + [4,5,6]
42/31: graph = n1 + n2 + n3 + n4 + n5 + n6
42/32: graph
42/33: graph > lista.txt
42/34: graph foo  > lista.txt
42/35: %store graph foo  > lista.txt
42/36: %store graph   > lista.txt
42/37: echo
42/38: toch lista.txt
42/39: touch lista.txt
42/40: %store graph
42/41: %store graph >a.txt
42/42: cat a.txt
43/1:
with open('full_list.txt') as f: 
    content = f.readlines()
43/2: content
43/3: content = set(content)
43/4: content
43/5: len(content)
43/6: %store content >clear_list.txt
43/7: f = open('nova_lista.txt', 'w')
43/8:
for i in range(content): 
    f.write(content[i])
43/9: content = list(content)
43/10: content[10]
43/11: r(content[10])
43/12:
print( r(content[10])
)
43/13: print( (content[10]))
43/14:
for i in range(content): 
    f.write(content[i])
43/15:
for i in range(len(content)): 
    f.write(content[i])
44/1: import matplotlib.pyplot as plt
44/2: import seaborn as sns
44/3: import pandas as pd
44/4: import numpy as np
44/5: df = pd.read_excel('HIST_PAINEL_COVIDBR_27ago2020.xlsx')
44/6: df.head()
44/7: df['municipio']
44/8: df['municipio']['Montes Claros']
44/9: df['municipio']
44/10: print(df.dtypes)
44/11: df.corr()
44/12:
df[['Recuperadosnovos' , 'semanaEpi']].corr(0
)
44/13: df[['Recuperadosnovos' , 'semanaEpi']].corr()
44/14: df[['Recuperadosnovos' , 'semanaEpi']].corr()
44/15: sns.regplot(x = "Recuperadosnovos", y = "semanaEpi", data = df)
44/16: plt.ylin(0, )
44/17: plt.ylim(0, )
44/18: corelation  = sns.regplot(x = "Recuperadosnovos", y = "semanaEpi", data = df)
44/19: sns
44/20: plt.show()
44/21: sns.set(color_codes = True)
44/22: plt.show()
44/23: sns.regplot(x = "Recuperadosnovos", y = "semanaEpi", data = df)
44/24: plt.show()
44/25: sns.regplot(x = "Recuperadosnovos", y = "semanaEpi", color = 'g', data = df)
44/26: plt.show()
44/27: sns.regplot(x = "Recuperadosnovos", y = "semanaEpi", color = 'g', data = df, x_estimator = np.mean)
44/28: plt.show()
44/29: df.describe()
44/30: df.describe(include=['object'])
44/31: df.dtypes
44/32: df['municipio'].where(df['municipio'] == 'Montes Claros')
44/33: df['municipio'].where(df['codmun'] == '314330')
44/34: moc  = df['municipio'].where(df['codmun'] == '314330')
44/35: moc
44/36: print(moc)
44/37: set(moc)
44/38: moc
44/39: moc[406898]
44/40: moc  = df[['municipio','codmun','estado', 'semanaEpi', 'casosAcumulado', 'casosNovos', 'obitosNovos']].where(df['codmun'] == '314330')
44/41: moc
44/42: moc['municipio']
44/43: moc['municipio'][1]
44/44: moc['municipio'][406935]
44/45: moc[406935]
44/46: moc['regiao'][406935]
44/47: moc['regiao']
44/48: moc['regiao']
44/49: moc['municipio']
44/50: moc
44/51: moc.dropna()
44/52: moc  = df[['municipio','codmun','estado', 'semanaEpi', 'casosAcumulado', 'casosNovos', 'obitosNovos']].where(df['codmun'] != '314330')
44/53: moc
44/54: moc.dropna()
44/55: moc  = df[['municipio','codmun','estado', 'semanaEpi', 'casosAcumulado', 'casosNovos', 'obitosNovos']].where(df['codmun'] != '314330)
44/56: moc  = df[['municipio','codmun','estado', 'semanaEpi', 'casosAcumulado', 'casosNovos', 'obitosNovos']].where(df['codmun'] == 314330)
44/57: moc
44/58: moc.dropna()
44/59: moc
44/60: moc  = moc.dropna()
44/61: moc
44/62: moc.corr()
44/63: sns.regplot( x = "semanaEpi", y = "obitosNovos", data = moc)
44/64: plt.show()
44/65: sns.regplot( x = "casosNovos", y = "obitosNovos", data = moc)
44/66: plt.show()
44/67: moc[['casosNovos', 'obitosNovos']].corr()
44/68: %history
44/69: %history > history.txt
44/70: cat history.txt
44/71: ls
44/72: ls
44/73: touch history.txt
44/74: %history
44/75: np.random.seed(19680881)
44/76: N = 50
44/77: x  = set(moc['casosAcumulados'])
44/78: moc['casosAcumulados']
44/79: moc.dtypes
44/80: x  = set(moc['casosAcumulados'])
44/81: x  = set(moc['casosAcumulado'])
44/82: moc.dtypes
44/83: y = set(moc['semanaEpi'])
44/84: colors = np.random.rand(N)
44/85: area = (30 * np.random.rand(N)) ** 2
44/86: plt.scatter(x,y,s = area, c = colors, alpha = 0.5)
44/87: y = list(moc['semanaEpi'])
44/88: x  = list(moc['casosAcumulado'])
44/89: plt.scatter(x,y,s = area, c = colors, alpha = 0.5)
44/90: len(y)
44/91: len(x)
44/92: s = range(y)
44/93: s = range(len(y))
44/94: plt.scatter(x,y,s = area, c = colors, alpha = 0.5)
44/95: len(s)
44/96: s
44/97: print(s)
44/98: s = list(s)
44/99: s
44/100: plt.scatter(x,y,s = area, c = colors, alpha = 0.5)
45/1: ls
45/2: import padas as pd
46/1: pip install pandas
46/2: import pandas as pd
46/3: ls
46/4: df = pd.read_csv('HIST_PAINEL_COVIDBR_27ago2020.xlsx')
46/5: ls
46/6: df = pd.read_csv('data_moc.csv')
46/7: df
46/8: df = pd.read_excel('HIST_PAINEL_COVIDBR_27ago2020.xlsx')
46/9: df.dtypes()
46/10: df.dtypes
46/11: df.dtypes
46/12: df[['populacaoTCU2019']].where(df['codmun'] == 314330)
46/13: df[['populacaoTCU2019']].where(df['codmun'] == 314330).dropna()
46/14: df.dtypes
46/15: moc = df[['regiao','estado','municipio','coduf','codmun','codRegiaoSaude','nomeRegiaoSaude', 'data','semanaEpi','populacaoTCU2019','casosAcumulado','casosNovos','obitosAcumulado','obitosNovos','Recuperadosnovos','emAcompanhamentoNovos']].where(df['codmun'] == 314330)
46/16: moc
46/17: moc.dropna()
46/18: moc = moc.dropna()
46/19: moc
46/20: moc.head()
46/21: moc = df[['regiao','estado','municipio','coduf','codmun','codRegiaoSaude','nomeRegiaoSaude', 'data','semanaEpi','populacaoTCU2019','casosAcumulado','casosNovos','obitosAcumulado','obitosNovos','Recuperadosnovos','emAcompanhamentoNovos']].where(df['codmun'] == 314330)
46/22: moc
46/23: moc = df[['regiao','municipio','codmun', 'data','semanaEpi','populacaoTCU2019','casosAcumulado','casosNovos','obitosAcumulado','obitosNovos','Recuperadosnovos','emAcompanhamentoNovos']].where(df['codmun'] == 314330)
46/24: moc
46/25: moc.dropna()
46/26: moc = df[['municipio', 'data','semanaEpi','populacaoTCU2019','casosAcumulado','casosNovos','obitosAcumulado','obitosNovos','Recuperadosnovos','emAcompanhamentoNovos']].where(df['codmun'] == 314330)
46/27: moc
46/28: moc.dropna()
46/29: moc = df[['municipio', 'semanaEpi','populacaoTCU2019','casosAcumulado','casosNovos','obitosAcumulado','obitosNovos','Recuperadosnovos','emAcompanhamentoNovos']].where(df['codmun'] == 314330)
46/30: moc
46/31: moc.dropna()
46/32: moc = df[['municipio', 'semanaEpi','casosAcumulado','casosNovos','obitosAcumulado','obitosNovos','Recuperadosnovos','emAcompanhamentoNovos']].where(df['codmun'] == 314330)
46/33: moc.dropna()
46/34: moc = df[['municipio', 'semanaEpi','casosAcumulado','casosNovos','obitosAcumulado','obitosNovos','Recuperadosnovos','emAcompanhamentoNovos']].where(df['codmun'] == 314330)
46/35: moc
46/36: moc.dropna()
46/37: %paste
46/38: moc
46/39: moc.dropna()
46/40: moc = df[['municipio', 'semanaEpi','casosAcumulado','casosNovos','obitosAcumulado','obitosNovos','Recuperadosnovos','emAcompanhamentoNovos']].where(df['codmun'] == 314330)
46/41: %paste
46/42: moc.dropna()
46/43: %paste
46/44: df.dtypes
46/45: %paste
46/46: moc
46/47: moc.dropna
46/48: moc.dropna()
46/49: %paste
46/50: %paste
46/51: moc
46/52: moc.dropna()
46/53: %paste
46/54: moc.dropna()
46/55: recup = df['Recuperadosnovos'].where(df['codmun'] == 314330)
46/56: recup
46/57: recup.dropna()
46/58: recup = df[['Recuperadosnovos']].where(df['codmun'] == 314330)
46/59: recup.dropna()
46/60: pip install jupyter
47/1:
# importando modulos
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
47/2:
#importando data-set bruto
df = pd.read_excel('HIST_PAINEL_COVIDBR_27ago2020.xlsx')
47/3:
# mostrando os dados disponíveis no data-set bruto
df.dtypes
47/4:
# filtrando dados para a região de Montes Claros
moc = df[['municipio','codmun','estado', 'semanaEpi', 'casosAcumulado', 'casosNovos', 'obitosNovos', 'obitosAcumulado']].where(df['codmun'] == 314330)

#retirando dados "perdidos" NaN
moc = moc.dropnan()
moc
47/5:
# filtrando dados para a região de Montes Claros
moc = df[['municipio','codmun','estado', 'semanaEpi', 'casosAcumulado', 'casosNovos', 'obitosNovos', 'obitosAcumulado']].where(df['codmun'] == 314330)

#retirando dados "perdidos" NaN
moc = moc.dropna()
moc
47/6:
# correlação entre dados - primeiro passo para analise com regressão linear

moc.corr()
47/7: moc[['casosAcumulado','semanaEpi']].corr()
47/8:
sns.regplot(x='casosAcumalado', y = 'semanaEpi', data= moc)
plt.ylim(0, )
47/9:
sns.regplot(x='casosAcumulado', y = 'semanaEpi', data= moc)
plt.ylim(0, )
47/10: moc[['obitosAcumulado', 'semanaEpi', 'casosAcumulado', 'obitosNovos', 'casosNovos']].corr()
47/11: sns.boxplot(x = "obitosAcumulado", y = "casosAcumulado", data = moc)
47/12: sns.regplot(x = "obitosAcumulado", y = "casosAcumulado", data = moc)
47/13: sns.regplot(x = 'semanaEpi', y = 'casosNovos', data = moc)
47/14: moc.describe()
47/15: moc.describe(include=['object'])
47/16: moc['semanaEpi'].value_counts()
47/17: moc['obitosNovos'].value_counts()
47/18: moc['obitosNovos'].value_counts().to_frame()
47/19:
obitos = moc['obitosNovos'].value_counts().to_frame()
obitos.index.name = "Frequencia_obitos"
obitos
47/20: obitos['obitosNovos']
47/21: obitos['Frequencia_obitos']
47/22: obitos_pivot  = obtios.pivot(indesx="Frequencia_obitos", columuns = "ObitosNovos")
47/23: obitos_pivot  = obitos.pivot(indesx="Frequencia_obitos", columuns = "ObitosNovos")
47/24: obitos_pivot  = obitos.pivot(index="Frequencia_obitos", columuns = "ObitosNovos")
47/25: obitos_pivot  = obitos.pivot(index="Frequencia_obitos", columuns = "obitosNovos")
47/26: obitos_pivot  = obitos.pivot(index="obitosNovos", columuns = "Frequencia_obitos")
47/27: obitos_pivot  = obitos.pivot(index="obitosNovos")
47/28: obitos_pivot   = obitos.fillna(0)
47/29: obitos_pivot   = obitos.fillna(0)
47/30:
obitos_pivot   = obitos.fillna(0)
obitos_pivot
47/31: # pivotar os dados de obitos
48/1: ls
48/2: import pandas
48/3: import pandas as pd
50/1: import pandas as pd
50/2: LS
50/3: ls
50/4: cd Developer/TCC/code
50/5: df = pd.read_excel('HIST_PAINEL_COVIDBR_27ago2020.xlsx')
50/6: df.dtypes
50/7: df.head()
50/8: regiao = df['regiao']
50/9: regiao
50/10: regiao.describe()
50/11: regiao.value_counts()
50/12:
# mostrando os dados disponíveis no data-set bruto
df.dtypes
51/1:
import pandas as pd

df =  pd.read_excel('HIST_PAINEL_COVIDBR_27ago2020.xlsx'); 

df['regiao'].describe()
df['regiao'].value_counts()
51/2: df['regiao'].describe()
51/3:
nordeste = df[['casosAcumulado', 'populacaoTCU2019','casosNovos', 'obitosAcumulado', 'obitosNovos']].where(
    df['regiao'] == 'Nordeste')
nordeste.head(30)
51/4:
nordeste = df[['casosAcumulado', 'populacaoTCU2019','casosNovos', 'obitosAcumulado', 'obitosNovos']].where(
    df['regiao'] == 'Nordeste')
nordeste.dropna()
51/5:
nordeste = df[['casosAcumulado', 'populacaoTCU2019','casosNovos', 'obitosAcumulado', 'obitosNovos']].where(
    df['regiao'] == 'Nordeste')
nordeste = nordeste.dropna()
nordeste.head(30)
51/6:
nordeste = df[['semanaEpi','casosAcumulado', 'populacaoTCU2019','casosNovos', 'obitosAcumulado', 'obitosNovos']].where(
    df['regiao'] == 'Nordeste')
nordeste = nordeste.dropna()
nordeste.head(30)
51/7:
nordeste = df[['semanaEpi','casosAcumulado', 'populacaoTCU2019','casosNovos', 'obitosAcumulado', 'obitosNovos']].where(
    df['regiao'] == 'Nordeste')
nordeste = nordeste.dropna()
nordeste.head(25:50)
51/8:
nordeste = df[['semanaEpi','casosAcumulado', 'populacaoTCU2019','casosNovos', 'obitosAcumulado', 'obitosNovos']].where(
    df['regiao'] == 'Nordeste')
nordeste = nordeste.dropna()
nordeste.head(50)
51/9: nordeste.corr()
51/10:
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.regplot(x="casosAcumulado", y="semanaEpi", data = nordeste)
51/11:
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.regplot(x="semanaEpi", y="casosAcumulado", data = nordeste)
51/12:
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.regplot(x="semanaEpi", y="casosAcumulado", data = nordeste)
plt.ylim(0, )
49/1:
sns.regplot(x='semanaEpi', y = 'casosAcumulado', data= moc)
plt.ylim(0, )
49/2:
# importando modulos
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
49/3:
#importando data-set bruto
df = pd.read_excel('HIST_PAINEL_COVIDBR_27ago2020.xlsx')
51/13: sns.regplot(x='semanaEpi', y = "casosNovos", data = nordeste)
49/4:
# filtrando dados para a região de Montes Claros
moc = df[['municipio','codmun','estado', 'semanaEpi', 'casosAcumulado', 'casosNovos', 'obitosNovos', 'obitosAcumulado']].where(df['codmun'] == 314330)

#retirando dados "perdidos" NaN
moc = moc.dropna()
moc
49/5:
sns.regplot(x='semanaEpi', y = 'casosAcumulado', data= moc)
plt.ylim(0, )
51/14: sns.hist(x='semanaEpi', y = "casosNovos", data = nordeste)
51/15: sns.distplot(x='semanaEpi', y = "casosNovos", data = nordeste)
51/16: sns.distplot(nordeste['casosNovos'])
51/17:
plot = nordeste['casosNovos'].hist(bins=1000,grid=False)
plt.xlabel("casos novos", fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.xscale('log')
51/18:
plot = nordeste['casosNovos'].hist(bins=1000,grid=False)
plt.xlabel("casos novos", fontsize=15)
plt.ylabel("Frequency",fontsize=15)
51/19:
plot = nordeste['casosNovos'].hist(bins=1000,grid=False)
plt.xlabel("casos novos", fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.xscale('log')
51/20:
plot = nordeste['casosNovos'].hist(bins=100,grid=True)
plt.xlabel("casos novos", fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.xscale('log')
51/21:
plot = nordeste['casosNovos'].hist(bins=10000,grid=True)
plt.xlabel("casos novos", fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.xscale('log')
51/22:
plot = nordeste['casosNovos'].hist(bins=1000,grid=True)
plt.xlabel("casos novos", fontsize=15)
plt.ylabel("frequência",fontsize=15)
plt.xscale('log')
51/23:
nordeste['casosNovos'].hist(bins=1000,grid=True)
plt.xlabel("casos novos", fontsize=15)
plt.ylabel("frequência",fontsize=15)
plt.xscale('log')
51/24: sns.distplot(nordeste['casosNovos'], kde=False, color='red', bins=100)
51/25: sns.distplot(nordeste['semanaEpi'], kde=False, color='red', bins=1000)
51/26: sns.distplot(nordeste['semanaEpi'], kde=True, color='red', bins=1000)
51/27:
nordeste['casosNovos'] = nordeste['casosNovos'].astype(int)
sns.distplot(nordeste['casosNovos'], kde=True, color='red', bins=1000)
51/28:
nordeste['casosNovos'] = nordeste['casosNovos'].astype(int)
nordeste.dtypes
51/29:
nordeste['casosNovos'] = nordeste['casosNovos'].astype(int)
nordeste.dtypes
nordeste['casosNovos'].hist(bins=1000,grid=True)
plt.xlabel("casos novos", fontsize=15)
plt.ylabel("frequência",fontsize=15)
plt.xscale('log')
51/30:
nordeste['casosNovos'] = nordeste['casosNovos'].astype(int)
nordeste.dtypes
51/31:
nordeste['casosAcumulado'] = nordeste['casosAcumulado'].astype(int)
nordeste['semanaEpi'] = nordeste['semanaEpi'].astype(int)
nordeste.dtypes
51/32: sns.regplot(x="semanaEpi", y="casosAcumulado", data = nordeste)
51/33: sns.regplot(x="semanaEpi", y="casosAcumulado", data = nordeste,  kde=True)
53/1: import pandas as pd
53/2: import numpy as np
53/3: import matplotlib.pyplot as plt
53/4: import seaborn as sns
53/5: data_url = 'http://bit.ly/2cLzoxH'
53/6: gapminder = pd.read_csv(data_url)
53/7: gapminder
53/8: gapminder.head(3)
53/9: gapminder.head(n= 3)
53/10: gapminder['lifeExp'].hist(bins = 100)
53/11: plt.show()
53/12: gapminder['lifeExp'].hist(bins = 10)
53/13: plt.show()
53/14: gapminder['country'].hist(bins = 10)
53/15: plt.show()
53/16: gapminder['country'].hist(bins = 100)
53/17: plt.show()
54/1: cd Downloads/
54/2: import pandas as pd
54/3: df = pd.read_excel('Teste Contabilidade e Custos.xlsx')
54/4: df
54/5: df.dropna()
54/6: df
54/7: df.dtypes
54/8: df.head(,include = 'object')
56/1: ls
56/2: ls
56/3: cd Developer/
56/4: ls
56/5: ls
56/6: cd TCC/
56/7: ls
56/8: cd code/
56/9: ls
56/10: import pandas as pd
56/11: df = pd.read_excel('HIST_PAINEL_COVIDBR_27ago2020.xlsx')
56/12: df.head(10)
56/13: brasil = df[['coduf','codmun','municipio']]
56/14: brasil
56/15: brasil = df[['coduf','codmun','municipio']]
56/16: brasil
56/17: brasil.dropna()
56/18: brasil = brasil.dropna()
56/19: brasil.to_dict()
56/20: municio = brasil.to_dict()
56/21: municio
56/22: brasil = df[['coduf','codmun','municipio', 'estado']]
56/23: brasil.dropna()
56/24: brasil = brasil.dropna()
56/25: municio = brasil.to_dict()
56/26: municio
56/27: pip install pickle
56/28: pip install pickle5
56/29: import pickle5 as pickle
56/30: filename = 'municipios'
56/31: outfile = open(filename, 'wb')
56/32: pickle.dump(municio,outfile)
56/33: outfile.close()
56/34: ls
56/35: cat municipios
59/1: import pickle5 as pickle
59/2: municipios = open('municipios', 'rb')
59/3: municipios = pickle.load(municipios)
59/4: municipios
59/5: import json
59/6: municipios
59/7: j = json.dumps(municipios)
59/8: j
59/9: file  = open('municipios.json', 'wb')
59/10: file.write(j)
59/11: file  = open('municipios.json', 'w')
59/12: file.write(j)
59/13: file.close()
60/1: import pandas as pd
60/2: df = pd.read_excel('HIST_PAINEL_COVIDBR_27ago2020.xlsx')
60/3: brasil = df[['coduf','codmun','municipio', 'estado']]
60/4: brasil = brasil.dropna()
60/5: brasil.to_csv('municipios.csv')
60/6: ls
60/7: brasil[0]
60/8: brasil = brasil.dropna()
60/9: municipios = brasil.to_dict()
60/10: set(municipios)
60/11: municipios.municipio
60/12: municipios.keys
60/13: municipios.keys()
60/14: municipios['municipio']
60/15: set(municipios['municipio'])
60/16: set(municipios['codmun'])
60/17: municipios['codemun']
60/18: municipios.keys()
60/19: municipios.codmun
60/20: municipios['codmun']
60/21: set(municipios['codmun'])
60/22: mun = set(municipios['codmun'])
60/23: mun
62/1: ls
62/2: cd Developer
62/3: ls
62/4: ls
62/5: cd TCC/
62/6: ls
62/7: cd code/
62/8: ls
62/9: import pickle5 as pickle
62/10: mun = pickle.load('municipios')
62/11: mun = open('municipios','r')
62/12: mun = pickle.load('municipios')
62/13: mun = pickle.load(mun)
62/14: mun = open('municipios','rb')
62/15: mun = pickle.load(mun)
62/16: mun
62/17: ls
62/18: mun
62/19: mun['RO']
62/20: mun[12]
62/21: mun.keys()
62/22: mun['municipio']
62/23: moc  = []
62/24: moc  = mun['municipio'] for _ in range(len(mun))
62/25: moc['codmun']
62/26: mun
62/27: mun['codmun']
62/28: lista  = x for x in (mun['codmun']) if x == 6333
62/29: x for x in 'matematica' if x in['a','e','i','o','u']
62/30: for x in 'matematica' if x in['a','e','i','o','u']
62/31: lista  = [x for x in 'matematica' if x in['a','e','i','o','u'] ]
62/32: lista
62/33: lista  = [x for x in (mun['codmun']) if x == 6333]
62/34: lista
62/35: lista  = [x for x in (mun['codmun']) if x == 9333]
62/36: lista
62/37: [x for x in (mun['codmun']) print(x)]
62/38: lista  = [x for x in (mun['codmun']) ]
62/39: lista
62/40: codmun  = [x for x in (mun['codmun']) ]
62/41: mun.keys()
62/42: name = [ x for x in (mun['municipio'])]
62/43: zip(name, codmun)
62/44: chaves  = zip(name, codmun)
62/45: chaves
62/46: print(chaves)
62/47: print(tuple(chaves))
62/48: name = [ x for x in (mun['municipio'])]
62/49: name
62/50: (mun['municipio'])
62/51: index  = mun['municipio']
62/52: ls
62/53: index
62/54: type(index)_
62/55: type(index)
62/56: index[i]
62/57: index[1]
62/58: index[6333]
62/59: index[6533]
62/60: ls
62/61: import pandas as pd
62/62: df = pd.read_excel('HIST_PAINEL_COVIDBR_27ago2020.xlsx')
62/63: df.head(30)
62/64: df.dtypes
62/65: regiao  = df.where( df['regiao'] )
62/66: regiao  = df.where( df['regiao'] != 'NaN' )
62/67: regiao
62/68: df
62/69: regiao
62/70: regiao  = df.where( df['regiao'] != 'Brasil' )
62/71: regiao
62/72: regiao = regiao.dropna()
62/73: regiao
62/74: regiao  = df.where( df['regiao'] != 'Brasil' )
62/75: ls
62/76: regiao.to_markdown('regiao.md')
62/77: pip install tabulate
62/78: regiao.to_markdown('regiao.md')
62/79: ls
62/80: br  = df.where( df['regiao'] == 'Brasil' )
62/81: br
62/82: df
62/83: br  = df[['']].where( df['regiao'] == 'Brasil' )
62/84: df.dtypes
62/85: br  = df[['data','semanaEpi','populacaoTCU2019','emAcompanhamentoNovos','obitosAcumulado','obitosNovos','casosNovos','casosAcumulado']].where( df['regiao'] == 'Brasil' )
62/86: bf
62/87: br
62/88: br.dropna()
62/89: br = br.dropna()
62/90: br
62/91: br['semanaEpi'].max()
62/92: br['casosNovos'].mean()
62/93: float(br['casosNovos'].mean(),2)
62/94: br.mean(axis = 0)
62/95: br.mean(axis = 0, skipna = True)
62/96: df['casosNovos'].mean()
62/97: df
62/98: br['casosNovos'].mean()
62/99: type(br['casosNovos'])
62/100: br.dtypes
62/101: br
62/102: br['casosNovos'].astype('int64')
62/103: br.head(20)
62/104: br.dtypes
62/105: br['casosNovos'].astype('int64')
62/106: br['casosNovos'].astype('int64').dtypes
62/107: br
62/108: br.dtypes
62/109: br['casosNovos'].astype(int)
62/110: br.dtypes
62/111: br['casosNovos']  = br['casosNovos'].astype(int)
62/112: br.dtypes
62/113: br['casosNovos'].mean()
62/114: br['casosNovos'].mean().round()
62/115: br['casosNovos'].mean().round(3)
62/116: sum(br['casosNovos'])
62/117: sum(br['casosNovos']) / len(br['casosNovos'])
62/118: br
62/119: br['semanaEpi'] = br['semanaEpi'].astype(int)
62/120: br.dtypes()
62/121: br.dtypes()
62/122: br.dtypes
62/123: br['semanaEpi'].count()
62/124: len(br['semanaEpi'])
62/125: br
62/126: br.head(8)
62/127: br['semanaEpi'] / 7
62/128: semana  = br['semanaEpi'] / 7
62/129: semana
62/130: type(semana)
62/131: len(semana)
62/132: 35 ¹ 7
62/133: 35 / 7
62/134: br / 7
62/135: br['data'].drop()
62/136: df.drop('data')
62/137: br.dtypes
62/138: br.drop('data', axis = 1)
62/139: br = br.drop('data', axis = 1)
62/140: br
62/141: br / 7
62/142: br
62/143: len(br)
62/144: 131 / 7
62/145: frame  = [ x for x in range(2) br['semanaEpi'].where(br['semanaEpi'] == x)]
62/146: br['semanaEpi']
62/147: semana0 = br.where(br['semanaEpi'] == 0)
62/148: semana0
62/149: semana0.dropna()
62/150: br[['semanaEpi']].where(br['semanaEpi'] == 0)
62/151: frame  = br[['semanaEpi']].where(br['semanaEpi'] == 0)
62/152: frame.head(12)
62/153: frame.head()
62/154: frame.dropna()
62/155: br.dtypes
62/156: br[['semanaEpi','casosNovos']].where(br['semanaEpi'] == 0)
62/157: br[['semanaEpi','casosNovos']].where(br['semanaEpi'] == 1)
62/158: br[['semanaEpi','casosNovos']].where(br['semanaEpi'] == 1).dropna()
62/159: br
62/160: br.head(1)
62/161: br[['semanaEpi','casosNovos']].where(br['semanaEpi'] == 17).dropna()
62/162: df.head(10)
62/163: df.head(20)
62/164: df.head(40)
62/165: df.head(50)
62/166: df[['seamanaEpi','casosNovos']].head(50)
62/167: df.dytpes
62/168: df.dtypes
62/169: df[['semanaEpi','casosNovos']].head(50)
62/170: br  = df[['data','semanaEpi','populacaoTCU2019','emAcompanhamentoNovos','obitosAcumulado','obitosNovos','casosNovos','casosAcumulado']].where( df['regiao'] == 'Brasil' )
62/171: br[['semanaEpi','casosNovos']].where(br['semanaEpi'] == 0).dropna()
62/172: br[['semanaEpi','casosNovos']].where(br['semanaEpi'] == 0)
62/173: br[['semanaEpi','casosNovos']].where(br['semanaEpi'] == 0).head(10)
62/174: br[['semanaEpi','casosNovos']].where(br['semanaEpi'] == 0).head(17)
62/175: df
62/176: br[['semanaEpi','casosNovos']].where(br['semanaEpi'] == 9).head(17)
62/177: br[['semanaEpi','casosNovos']].where(br['semanaEpi'] == 9).dropna().head(17)
62/178: br
62/179: ls
62/180: br.to_csv('brasil27ago.csv')
62/181: ls
62/182: brasil  = br.to_dict()
62/183: type(brasil)
62/184: brasil
62/185: file  = open('brasil_27ago_dic', 'wb')
62/186: pickle.dump(brasil)
62/187: pickle.dump(brasil,file)
62/188: file.close()
62/189: git
63/1: ls
63/2: cd Developer/
63/3: ls
63/4: ls
63/5: cd TCC/
63/6: ls
63/7: cd code/
63/8: ls
63/9: import pandas as pd
63/10: br = pd.read_csv('brasil27ago.csv')
63/11: br
63/12: br = br.dropna()
63/13: br
63/14: data  = br.data
63/15: data
63/16: type(data[1])
63/17: type(data)
63/18: data[1]
63/19: data.head(3)
63/20: data[55]
63/21: type(data[50])
63/22: type(data[55])
63/23: data[55]
63/24: data[55].split()
63/25: data[55].split('-')
63/26: data[55].split('-')[1]
63/27: [ x for x in data]
63/28: [x for x in data print(x.split('-')[1])]
63/29: [x for x in data x.split('-')[1]]
63/30: [for x in data x.split('-')[1]]
63/31: [ x for x in data ]
63/32: lista  = [ x for x in data x ]
63/33: lista  = [ x for x in data print(x) ]
63/34: new_data = pd.Series()
63/35: new_data.add(12)
63/36: new_data
63/37: new_data.head()
63/38: new_data[1]
63/39: new_data[0]
63/40: new_data.add({ 0: 'kaio'})
63/41: new_data[0]
63/42: new_data[0] = 'eita'
63/43: new_data.append(123)
63/44: new_data = pd.Series([])
63/45: new_data.append([1,2,1])
63/46: lista  = []
63/47:
for i in data:
    print(i)
63/48: data[55].split('-')[1]
63/49: lista
63/50:
for i in data: 
    lista.append(i.split('-')[1])
63/51: lista
63/52: len(lista)
63/53: new_data = pd.Series(lista, index = range(131))
63/54: new_data
63/55: br
63/56: br['mes'] = new_data
63/57: br
63/58: new_data = pd.Series(lista, index = range(54,184))
63/59: new_data = pd.Series(lista, index = range(54,185))
63/60: new_data
63/61: br['mes'] = new_data
63/62: br
63/63: br.to_csv('brasil27agoMes.csv')
64/1: ls
64/2: ls
64/3: ls
64/4: ls
64/5: cd Downloads
64/6: ls
64/7: import pandas as pd
64/8: df  = pd.read_excel('HIST_PAINEL_COVIDBR_06set2020.xlsx')
64/9: df
64/10: df.dtypes
64/11: query = ['região','data','semanaEpi','casosAcumulado','obitosAcumulado','obitosNovos','Recuperadosnovos','emAcompanhamentoNovos']
64/12: br = df[query].where(df['reiao'] == 'Brasil')
64/13: br = df[query].where(df['regiao'] == 'Brasil')
64/14: query = ['regiao','data','semanaEpi','casosAcumulado','obitosAcumulado','obitosNovos','Recuperadosnovos','emAcompanhamentoNovos']
64/15: br = df[query].where(df['regiao'] == 'Brasil')
64/16: br
64/17: br['regiao'][920903]
64/18: br['regiao'][920902]
64/19: br['regiao'][920900]
64/20: br = br.dropna()
64/21: br
64/22: ls
64/23: br
64/24: ls
64/25: cd ..
64/26: ls
64/27: cd Developer/TCC/
64/28: ls
64/29: cd code/
64/30: ls
64/31: br.to_csv('brasil06set.csv')
64/32: ls
64/33: lista  = []
64/34:
for i in data: 
    lista.append(i.split('-')[1])
64/35: data  = br['data']
64/36:
for i in data: 
    lista.append(i.split('-')[1])
64/37: br['data']
64/38:
for i in data: 
    lista.append(srt(i).split('-')[1])
64/39:
for i in data: 
    lista.append(str(i).split('-')[1])
64/40: lista
64/41: len(lista)
64/42: br['mes'] = lista
64/43: br
64/44: br.to_csv('brasil06setMes.csv')
64/45: ls
65/1: cd ..
66/1: ls
66/2: cd Downloads/
66/3: cat
66/4: cat jsonformatter.txt
66/5: cp jsonformatter.txt issues.csv
66/6: ls
66/7: cat issues.csv
67/1: import numpy as np
67/2: import tensorflow as tf
67/3: from tensorflow import keras
67/4: from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
67/5: training_data = np.array([["This is the 1st sample."], ["And heres's the 2nd sample."]])
67/6: vectorizer = TextVectorization(output_mode="int")
67/7: vectorizer
67/8: vectorizer.adapt(training_data)
67/9: integer_data = vectorizer(training_data)
67/10: integer_data
68/1: import pandas as pd
68/2: ls
69/1: cd Developer/Estudos/
69/2: cd Kaggle/
69/3: ls
69/4: import numpy as np
69/5: import pandas as pd
69/6: pip install plotly
69/7: import plotly.offline as py
69/8: import plotly.graph_obj as go
69/9: import plotly.graph_objs as go
69/10: import seaborn as sns
69/11: import matplotlib.pyplot as plt
69/12: py.init_notebook_mode(connected=False)
69/13: ls
69/14: import os
69/15:
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
69/16: df = pd.read_csv('Glassdoor Gender Pay Gap.csv')
69/17: df.head(12)
69/18: df.shape
69/19: df.dtypes
69/20: df.describe()
69/21: femaleDs = df[df['Gender'] == 'Female'].loc[:,['BasePay','JobTitle','Education', 'Seniority']]
69/22: maleDs = df[df['Gender'] == 'Male'].loc[:,['BasePay','JobTitle','Education', 'Seniority']]
69/23: print(' Há {} pessoas do sexo feminino e {} do sexo masculino', format( len(femaleDs.index), len(maleDs.index)))
69/24: print(' Há {} pessoas do sexo feminino e {} do sexo masculino'.format( len(femaleDs.index), len(maleDs.index)))
69/25: # desvio padrão
69/26: print('O desvio padrão da BasePay para o sexo feminino é de: {} e  para o masculino: {}'.format(np.std(femaleDs.BasePay), np.std(maleDs.BasePlay)))
69/27: print('O desvio padrão da BasePay para o sexo feminino é de: {} e  para o masculino: {}'.format(np.std(femaleDs.BasePay), np.std(maleDs.BasePay)))
69/28:
def DistPlot(x, color):
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    sns.distplot(x, bins=100, color=color)
69/29: DistPlot(femaleDs.BasePay, 'magenta')
69/30: plt.show()
69/31: DistPlot(maleDs.BasePay, 'pink')
69/32: plt.show()
69/33: DistPlot(maleDs.BasePay, 'gray')
69/34: DistPlot(maleDs.BasePay, 'gray')
69/35: plt.show()
69/36: print(' O maior salário para o sexo feminino é de: ${} e para o masculino: ${}'.format(femaleDs.BasePay.max(), maleDs.BasePay.max()))
69/37: # parei procurando outliers
69/38: boxFemalw  = go.Box(y = femaleDs.BasePay, name = 'Feminino' , marker = { 'color':'#e74c3c' })
69/39: boxFemale  = go.Box(y = femaleDs.BasePay, name = 'Feminino' , marker = { 'color':'#e74c3c' })
69/40: boxMale = go.Box( y = maleDs.BasePay, name = 'Masculino', marker = { 'color': '#212337' })
69/41: data = [ boxFemale, boxMale ]
69/42: layout = go.Layout( title = 'Dispersão de salário por gênero' , titlefont = { 'family': 'Arial', 'size' : 22, 'color':'#7f7f7f'} , xaxis = { 'title'  : 'Gênero' } , yaxis = { 'title' : 'Salário' }, paper_bgcolor= 'rgb( 243, 243, 243)', plot_bgcolor= 'rgb(243,243,243)')
69/43: fig = go.Figure(data=data, layout=layout)
69/44: py.iplot(fig)
69/45: py.plot()
69/46: plt.show()
69/47: fig.show()
69/48: sns.catplot( x = 'Gender', hue = 'Education', kind = 'count' , data=df)
69/49: plt.show()
69/50: df.JobTitle.unique()
69/51: sns.catplot(x = 'Gender', hue='JobTitle' , kind='count', data=df)
69/52: plt.show()
69/53: femaleDs  = femaleDs.loc[(femaleDs['JobTitle'] == 'Graphic Designer') | (femaleDs['JobTitle'] == 'Data Scientist')]
69/54: maleDs  = maleDs.loc[(maleDs['JobTitle'] == 'Graphic Designer') | (maleDs['JobTitle'] == 'Data Scientist')]
69/55: femaleDs.nlargest( 10, 'BasePay')
69/56: maleDs.nlargest( 10, 'BasePay')
69/57: df.head()
69/58: df.diff()
69/59: df.dtypes
69/60: df[df['Age'] >= 31].loc[:, ['BasePlay', 'JobTitle', 'Education', 'Seniority', 'Age']]
69/61: age = df[df['Age'] >= '31'].loc[:, ['BasePlay', 'JobTitle', 'Education', 'Seniority', 'Age']]
69/62: age = df[df['Age'] >= 31].loc[:, ['BasePlay', 'JobTitle', 'Education', 'Seniority', 'Age']]
69/63: age = df[df['Age'] >= 31]
69/64: age
69/65: age.nlargest( 10, 'BasePay')
69/66: dataScientist  = age.loc[(age['JobTitle'] == 'Data Scientist')]
69/67: dataScientist.nlargest(10, 'BasePay')
69/68: sns.catplot( x = 'Gender', hue = 'Education', kind='count', data= dataScientist)
69/69: plt.show()
69/70: len(dataScientist.index)
69/71: len(df.index)
69/72: ls
69/73: display jobs.png
69/74: ls
69/75: df
69/76: df['JobTitle'].unique()
69/77: len(df['JobTitle'].unique())
69/78: len(df['JobTitle'].unique())
69/79: df['JobTitle'].unique()
69/80: ls
69/81: cd Glassdoor\ Gender\ Pay\ Gap.csv
70/1: pip install sklearn
70/2: from sklearn.ensemble import RandomForestClassifier
70/3: clf  = RandomForestClassifier(random_state=0)
70/4: X = [[ 1,2,3],[11,12,13]]
70/5: y = [0,1]
70/6: clf.fit(X,y)
70/7: clf.predict(x)
70/8: clf.predict(X)
70/9: clf.predict([[4,5,6],[14,15,16]])
70/10: clf.predict([4,5,6])
70/11: clf.predict([4,5,6])
70/12: clf.predict([[14,15,16],[14,15,16]])
70/13: clf.predict([[14,15,16],[4,5,6]])
70/14: X = [[ 2,4,6],[1,3,5]]
70/15: y = [0,1]
70/16: clf.predict([[13,15,17],[4,8,6]])
70/17: from sklearn.preprocessing import StandardScaler
70/18: from sklearn.linear_model import LogisticRegression
70/19: from sklearn.pipeline
70/20: from sklearn.pipeline import make_pipeline
70/21: from sklearn.datasets import load_iris
70/22: from sklearn.model_selection import train_test_split
70/23: from sklearn.metrics import accuracy_score
70/24:
pipe  = make_pipeline( 
StandardScaler(), LogisticRegression( random_state=0 )
)
70/25: X, y = load_iris( return_X_y = True )
70/26: X_train, X_test, y_train, y_test = train_test_split( X, y, random_state = 0 )
70/27: pipe.fit(X_train, y_train)
70/28: accuracy_score(pipe.predict(X_test), y_test)
70/29: X_train
71/1: tiem
71/2: time
71/3: ls
72/1: import pandas as pd
72/2: df = pd.read_csv('complaints.csv')
72/3: df.shape
72/4: df.describe
72/5: df.head(10)
72/6: df.columns
72/7: df.dtypes()
72/8: df.dtypes
72/9: from fklearn.preprocessing.splitting import time_split_dataset
72/10: from fklearn.training.classification import nlp_logistic_classification_learner
72/11: from fklearn.validation.evaluators import fbeta_score_evaluator
72/12:
def load_data(path):
    df = pd.read_csv(path, usecols = [ "Product", "Consumer complaint narrative", "Date received", "Complaint ID"], parse_dates = [ "Date received"]).rename(columns = { "Product" : "product", "Consumer complaint narrative": "text", "Date received": "time" , "Complaint ID": "id" })
    df["target"] = (df["product"] == "Credit reporting, credit repair services, or other personal consumer reports").astype(int)
    return df.dropna()
72/13: df = load_data('complaints.csv')
72/14: df.head()
72/15: df.shape
72/16: train, holdout  = time_split_dataset( df, train_start_date="2017-01-01" , train_end_date="2018-01-01", time_column="time")
72/17: train, holdout = time_split_dataset(df, train_start_date="2017-01-01", train_end_date="2018-01-01", holdout_end_date="2019-01-01", time_column="time")
72/18: predic_fn, train_pred, logs  = nlp_logistic_classification_learner(train, text_feature_cols=["text"], target="target")
72/19: holdout_pred = predic_fn( holdout )
72/20: f1_score = fbeta_score_evaluator(holdout_pred)
72/21: f1_score
72/22: holdout_pred
72/23: cd
73/1: ls
73/2: cd Desktop/
73/3: ls
73/4: cd ..
73/5: ls
73/6: cd Developer/
73/7: ls
73/8: cd Ibm_curse/
73/9: ls
74/1: import pandas as pd
74/2: import numpy as pn
74/3: import numpy as np
74/4: path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
74/5: df = pd.read_csv(path)
74/6: df.head()
74/7: import matplotlib.pyplot as plot
74/8: import matplotlib.pyplot as plt
74/9: import seaborn as sns
74/10: df.dtypes
74/11: df.corr()
74/12: df['engine size']
74/13: df['engine-size']
74/14: relation  = ['bore', 'stroke','compression-ratio', 'horsepower']
74/15: df[relation]
74/16: df[relation].corr()
74/17: sns.regplot(x='engine-size',y='price', data = df)
74/18: plt.ylim(0, )
74/19: plt.show()
74/20: df[['engine-size','price']].corr()
74/21: sns.regplot(x='highway-mpg', y = 'price', data=df)
74/22: plot.show()
75/1: import pandas as pd
75/2: ls
75/3: cd Downloads/
75/4: df  = pd.read_csv('Cópia de train - train_post_competition.csv')
75/5: df
75/6: df.dtypes
75/7: df.describe
75/8: df[0]
75/9: df.dtypes
76/1: ls
76/2: cd Downloads/a
76/3: cd Downloads
76/4: import pandas as pd
76/5: df  = pd.read_csv('Cópia de train - train_post_competition.csv')
76/6: df.columns
76/7: df.rename(columns= = { 'gs://meishi-13f87-vcm/spectro-data/00044347.wav.jpg' : 'url' , 'Hi-hat' : 'result' } , inplace  = True )
76/8: df.rename(columns= { 'gs://meishi-13f87-vcm/spectro-data/00044347.wav.jpg' : 'url' , 'Hi-hat' : 'result' } , inplace  = True )
76/9: df.dtypes
76/10: names  = n for df.url print(n)
76/11: names  = n for n in df.url print(n)
76/12: names  = n for n in df.url
76/13: names  =[ n for n in df.url print(n)]
76/14: names  =[ n for n in df.url ]
76/15: names
76/16: names[0]
76/17: names[0].replace('meishi-13f87-vcm', 'automl_guide')
76/18: names[0]
76/19: ls
76/20: names  =[ n for n in df.url nmeishi-13f87-vcm]
76/21:
names  =[ n for n in df.url  n.replace('meishi-13f87-vcm', 'automl_guide')
]
76/22: names  =[ n for n in df.url return n.replace('meishi-13f87-vcm', 'automl_guide')]
76/23: names  =[ n for n in df.url print(n.replace('meishi-13f87-vcm', 'automl_guide'))]
76/24: names  =[ n for n in names n.replace('meishi-13f87-vcm', 'automl_guide')]
76/25:
def wraning(n): 
    return n.replace('meishi-13f87-vcm', 'automl_guide')
76/26: names  = [ n for n in names wraning(n) ]
76/27:
for i in names:
    i.replace('meishi-13f87-vcm', 'automl_guide')
76/28: names
76/29:
for i in names: 
    print(i)
76/30:
for i in range(names): 
    names[i].replace('meishi-13f87-vcm', 'automl_guide')
76/31:
for i in range(len(names)): 
    names[i].replace('meishi-13f87-vcm', 'automl_guide')
76/32: names
76/33: names[1].replace('meishi-13f87-vcm', 'automl_guide')
76/34: names
76/35: names[0:4]
76/36: _url = []
76/37:
for i in range(len(names)): 
   _url.append(names[i].replace('meishi-13f87-vcm', 'automl_guide'))
76/38: _url
76/39: url  = pd.Series(_url)
76/40: url
76/41: df.url
76/42: df.url  = url
76/43: df
76/44: ls
76/45: df.to_csv('auto_ml.csv')
76/46: ls
76/47: cat
76/48: cat auto_ml.csv
77/1: %run imports.py
77/2: capper_fn = capper( columns_to_cap=["income"] , precomputed_caps={ "income" : 50, 000})
77/3: capper_fn = capper( columns_to_cap=["income"] , precomputed_caps={ "income" : 50,000})
77/4: capper_fn = capper( columns_to_cap=["income"] , precomputed_caps={ "income" : 50,000})
77/5: capper_fn = capper(columns_to_cap=["income"], precomputed_caps={"income": 50,000})
77/6: capper_fn = capper(columns_to_cap=["income"], precomputed_caps={"income": 50,000})
77/7: capper_fn = capper(columns_to_cap=["income"], precomputed_caps={"income": 50.000})
77/8: regression_fn = linear_regression_learner(features=["income", "bill_amount"], target="spend")
77/9: p, df, log = regression_fn(training_data)
77/10: ranger_fn = prediction_ranger(prediction_min=0.0, prediction_max=20000.0)
77/11: p, df, log = regression_fn(training_data)
77/12: from fklearn.training.pipeline import build_pipeline
77/13:
learner = build_pipeline(capper_fn, regression_fn, ranger_fn)
predict_fn, training_predictions, logs = learner(train_data)
77/14: imrpot numpy as np
77/15: import numpy as np
77/16: import pandas as pd
77/17: import numpy.random as random
77/18: random.seed(150)
77/19: dates  = pd.DataFrame({'score_date': pd.date_range('2016-01-01','2016-12-31')})
77/20: dates
77/21: dates['key'] = 1
77/22: dates
77/23: dates.dtypes
77/24: dates.head(3)
77/25: ids = pd.DataFrame({'id' : np.arange(0,100)})
77/26: ids
77/27: ids['key'] = 1
77/28: idf
77/29: ids
77/30: data = pd.merge(ids, dates).drop('key', axis=1)
77/31: data
77/32: data[300]
77/33: data.head( 300: 331)
77/34: data.head[ 300:331]
77/35: data.head(300:331)
77/36: data.head(300)
77/37:
data['x1'] = 23 * random.randn(data.shape[0]) + 500
data['x2'] = 59 * random.randn(data.shape[0]) + 235
data['x3'] = 73 * random.randn(data.shape[0]) + 793  # Noise feature.

data['y'] = 0.37*data['x1'] + 0.97*data['x2'] + 0.32*data['x2']**2 - 5.0*data['id']*0.2 + \
            np.cos(pd.to_datetime(data['score_date']).astype(int)*200)*20.0

nan_idx = np.random.randint(0, data.shape[0], size=100)  # Inject nan in x1.
data.loc[nan_idx, 'x1'] = np.nan

nan_idx = np.random.randint(0, data.shape[0], size=100)  # Inject nan in x2.
data.loc[nan_idx, 'x2'] = np.nan
77/38: data.head()
77/39: data['y'][data['id'] == 0].plot()
77/40: from matplotlib import pyplot as plt
77/41: pip install matplotlib
77/42: from matplotlib import pyplot as plt
77/43: data['y'][data['id'] == 0].plot()
78/1: %run imports.py
78/2: import padas as pd
78/3: import pandas as pd
78/4: %run imports.py
78/5: %rum imports.py
78/6: %run imports.py
78/7: data.head(3)
78/8: data['y'][data['id']==0].plot()
78/9: plt.show()
78/10: from fklearn.preprocessing.splitting import space_time_split_dataset
78/11:
from fklearn.preprocessing.splitting import space_time_split_dataset

train_start = '2016-01-01'
train_end = '2016-06-30'
holdout_end = '2016-12-31'

split_fn = space_time_split_dataset(
    train_start_date=train_start,
    train_end_date=train_end,
    holdout_end_date=holdout_end,
    split_seed=50,
    space_holdout_percentage=.05,
    space_column='id',
    time_column='score_date',
)
78/12: train_set, intime_outspace_hdout, outime_inspace_hdout, outime_outspace_hdout = split_fn(data)
78/13: train_set.shape, intime_outspace_hdout.shape, outime_inspace_hdout.shape, outime_outspace_hdout.shape
78/14: FEATURES = ['x1', 'x2', 'x3']
78/15: TARGET = ['y']
78/16: from fklearn.training.imputation import imputer
78/17: my_imputer = imputer(columns_to_impute=FEATURES, impute_strategy='median')
78/18: from fklearn.training.transformation import standard_scaler
78/19: my_scaler = standard_scaler(columns_to_scale=FEATURES)
78/20:
from fklearn.training.regression import xgb_regression_learner

my_model = xgb_regression_learner(
    features=['x1', 'x2', 'x3'],
    target='y',
    prediction_column='prediction',
    extra_params={'seed': 139, 'nthread': 8},
)
78/21: from fklearn.training.transformation import ecdfer
78/22: my_ecdefer = ecdfer(prediction_column='prediction', ecdf_column='prediction_ecdf')
78/23: from fklearn.training.pipeline import build_pipeline
78/24: my_learner = build_pipeline(my_imputer, my_scaler, my_model, my_ecdefer)
78/25: (prediction_function, _, logs) = my_learner(train_set)
79/1: cd Downloads/
79/2: cd HIST_PAINEL_COVIDBR_05out2020/
79/3: import pandas as pd
79/4: ls
79/5: df  = pd.read_csv('HIST_PAINEL_COVIDBR_05out2020.csv')
79/6: df  = pd.read_excel('HIST_PAINEL_COVIDBR_05out2020.csv')
79/7: df -h
79/8: ls
79/9: cd ..
79/10: ls
79/11: df  = pd.read_excel('HIST_PAINEL_COVIDBR_05out2020.xlsx')
79/12: ls
79/13: df  = pd.read_csv('HIST_PAINEL_COVIDBR_05out2020 - HIST_PAINEL_COVIDBR_05out2020.csv')
79/14: df.head
79/15: df.dtypes
79/16: df  = pd.read_csv('hist.csv')
79/17: df.head
79/18: df.columns
79/19: df['regiao']
79/20: df['regiao'].unique()
79/21: df['regiao']['Sudeste']
79/22: df['regiao'][['Sudeste']]
79/23: df['regiao'].where[df['regiao'] == 'Sudeste']
79/24: df['regiao'].where(df['regiao'] == 'Sudeste')
79/25: df['regiao'].where(df['regiao'] == 'Sudeste').dropan()
79/26: df['regiao'].where(df['regiao'] == 'Sudeste').dropna()
79/27: sudeste = df['regiao'].where(df['regiao'] == 'Sudeste').dropna()
79/28: ls
79/29: df.to_csv('hist_05_oct.csv')
80/1: ls
80/2: cd Developer/TCC/
80/3: ls
80/4: cd code/
80/5: ls
80/6: ls
80/7: mkdir linear_regretion
80/8: cd linear_regretion/
80/9: jupter
80/10: jupyter
81/1: ls
82/1: from sklearn import linear_model
82/2: req = linear_model.LinearRegression()
82/3: reg.fit([[0,0],[1,1],[2,2]],[0,1,2])
82/4: req.fit([[0,0],[1,1],[2,2]],[0,1,2])
82/5: req.coef_
82/6: req.predict([[3,3]])
82/7: req.predict([[3,2]])
82/8: req.predict([[2,2]])
82/9: req.predict([[1,1]])
82/10: req.predict([[4,4]])
82/11: req.predict([[5,4]])
82/12: req.predict([[1,1]])
82/13: reg = linear_model.Lasso(alpha = 0.1)
82/14: reg.fit([[0,0],[1,1]],[0,1])
82/15: reg.predict([[1,1]])
82/16: reg.predict([[0,1]])
82/17: reg.predict([[0,0]])
82/18: reg.predict([[1,0]])
82/19: reg.predict([[1,2]])
82/20: reg.predict([[2,2]])
82/21:
import numpy as np
import sklearn.neural_network

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([0,1,1,0])

model = sklearn.neural_network.MLPClassifier(
                activation='logistic',
                max_iter=100,
                hidden_layer_sizes=(2,),
                solver='lbfgs')
model.fit(inputs, expected_output)
print('predictions:', model.predict(inputs))
82/22: %rum main.py
82/23: %run main.py
82/24: model.predict([0,0])
82/25: model.predict([[0,0],[1,1],[0,1],[1,0]])
82/26: model.predict([[0,1],[0,0],[1,0],[1,1]])
82/27: import Mode from main
82/28: from main import Mode
82/29: Mode.train()
82/30: from main import Mode
82/31: Mode.train()
82/32: Mode.train
82/33: Mode.test
82/34: from main import Model
82/35: from main import Model
82/36: ls
82/37: ls
82/38: from main import Model
82/39: from main import Mode
83/1: from main import Model
83/2: model  = Model.train()
83/3: model  = Model
83/4: model
83/5: model.train()
83/6: from teste import Teste
83/7: teste = Teste
83/8: Teste.show()
83/9: Teste.show(eita)
83/10: Teste.show
83/11: from teste import Teste
83/12: teste  = Teste(3)
84/1:
curl 'https://mobileapps.saude.gov.br/esus-vepi/files/unAFkcaNDeXajurGB7LChj8SgQYS2ptm/552009865d6963a691f1a44e47b9f1f0_HIST_PAINEL_COVIDBR_10out2020.csv' \
  -H 'authority: mobileapps.saude.gov.br' \
  -H 'user-agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Mobile Safari/537.36' \
  -H 'accept: */*' \
  -H 'origin: https://covid.saude.gov.br' \
  -H 'sec-fetch-site: same-site' \
  -H 'sec-fetch-mode: cors' \
  -H 'sec-fetch-dest: empty' \
  -H 'referer: https://covid.saude.gov.br/' \
  -H 'accept-language: pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7' \
  --compressed
84/2: import pandas as pd
84/3: url  = " https://mobileapps.saude.gov.br/esus-vepi/files/unAFkcaNDeXajurGB7LChj8SgQYS2ptm/552009865d6963a691f1a44e47b9f1f0_HIST_PAINEL_COVIDBR_11out2020.csv"
84/4: df = pd.read_csv(url)
84/5: pip install fsspec
84/6: url  = "https://mobileapps.saude.gov.br/esus-vepi/files/unAFkcaNDeXajurGB7LChj8SgQYS2ptm/552009865d6963a691f1a44e47b9f1f0_HIST_PAINEL_COVIDBR_11out2020.csv"
84/7: df = pd.read_csv(url)
84/8: ls
84/9: url  = "https://mobileapps.saude.gov.br/esus-vepi/files/unAFkcaNDeXajurGB7LChj8SgQYS2ptm/552009865d6963a691f1a44e47b9f1f0_HIST_PAINEL_COVIDBR_10out2020.csv"
84/10: df = pd.read_csv(url)
84/11: ls
84/12: cd Downloads/
84/13: ls
84/14: cat spotify & google-chrome
84/15: cat spotify & google-chrome
84/16: cat HIST_PAINEL_COVIDBR_10out2020.csv
84/17: ls
84/18: cat HIST_PAINEL_COVIDBR_10out2020.csv | tr ";" "," > 11_out2020.csv
84/19: df = pd.read_csv("11_out2020.csv")
84/20: df = pd.read_csv(url, error_bad_lines=False)
84/21: df
84/22: df.head
84/23: df = pd.read_csv("11_out2020.csv ", error_bad_lines=False)
84/24: df = pd.read_csv("11_out2020.csv", error_bad_lines=False)
84/25: df
84/26: df.dtypes
84/27: df[['codmun','município']]
84/28: df[['codmun','municipio']]
84/29: df[['codmun','municipio']].dropna()
84/30: df[['codmun','municipio']].dropna().unique()
84/31: df[['codmun']].dropna().unique()
84/32: df[['codmun']].dropna().unique
84/33: mun  = df[['codmun']].dropna()
84/34: type(mun)
84/35: mun
84/36: mun  = set(mun)
84/37: mun
84/38: mun  = df[['codmun']].dropna()
84/39: mun  = set(mun['codmun'])
84/40: mun
84/41: names  = []
84/42: df = df[['municipio', 'codmun']].where(df['codmun'] in mun)
84/43: mun = list(mun)
84/44: df = df[['municipio', 'codmun']].where(df['codmun'] in mun)
84/45: df = df[['municipio', 'codmun']].where(df['codmun'] in mun == True)
84/46: mun[1]
84/47: df = df[['municipio', 'codmun']].where(df['codmun'] == mun[1])
84/48: df
84/49: df.dropna()
84/50: df
84/51: df.dropna(inplace=True)
84/52: df
84/53: mun[1]
84/54: df = pd.read_csv("11_out2020.csv ", error_bad_lines=False)
84/55: df = pd.read_csv("11_out2020.csv", error_bad_lines=False)
84/56: df['municipio'].unique()
84/57: names  = df['municipio'].unique()
84/58: code = df['codmun'].unique()
84/59: len(names)
84/60: len(code)
84/61: code.dropna()
84/62: code
84/63: code  = pd.Series(code)
84/64: code.dropna()
84/65: code.dropna(inplace= True)
84/66: code
84/67: len(names)
84/68: df.columns
84/69: index = df.columns
84/70: index
84/71: index[1]
84/72: index[0]
84/73: index[0:5]
84/74: index[0:6]
84/75: index[6]
84/76: index[8]
84/77: index[9]
84/78: index[10]
84/79: index[19]
84/80: len(index)
84/81: index[17]
84/82: type(index)
84/83: index = list(index)
84/84: index
84/85: brasil = df[index].where(df['regiao'] == 'brasil')
84/86: brasil
84/87: brasil.dropna()
84/88: df
84/89: brasil = df[index].where(df['regiao'] == 'Brasil')
84/90: brasil
84/91: brasil.dropna(inplace=True)
84/92: brasil
84/93: brasil = df[index].where(df['regiao'] == 'Brasil')
84/94: brasil
84/95: brasil['regiao'].unique()
84/96: brasil.head(40)
84/97: brasil.head(60)
84/98: brasil.head(90)
84/99: brasil.head(100)
84/100: brasil.head(190)
84/101: brasil.head(690)
84/102: brasil = df[index].where(df['regiao'] == 'Brasil')
84/103: brasil.drop_duplicates()
84/104: brasil = df[index].where(df['regiao'] == 'Brasil' || df['regiao'] != 'NaN')
84/105: brasil = df[index].where(df['regiao'] == 'Brasil' and df['regiao'] != 'NaN')
84/106: brasil = df[index].where(df['regiao'] == 'Brasil' and pd.isna(df['regiao']))
84/107: brasil = df[index].where(df['regiao'] == 'Brasil' )
84/108: brasil = brasil[index].where(pd.isna(df['regiao']))
84/109: brasil
84/110: brasil = df[index].where(df['regiao'] == 'Brasil' )
84/111: brasil = brasil[index].where(!pd.isna(df['regiao']))
84/112: brasil = brasil[index].where(pd.isna(df['regiao']) == False  )
84/113: brasil
84/114: brasil = brasil[index].where(!pd.isna(df['regiao']))
84/115: brasil = df[index].where(df['regiao'] == 'Brasil' )
84/116: brasil = brasil[index].where(pd.isna(df['regiao']) == True)
84/117: brasil
84/118: brasil.dropna()
84/119: brasil = df[index].where(df['regiao'] == 'Brasil' )
84/120: brasil.describe()
84/121: brasil.dropna()
84/122: brasil.fillna()
84/123: brasil
84/124: brasil.dropna(subset=['regiao'])
84/125: brasil.dropna(subset=['regiao'], inplace=True)
84/126: brasil
84/127: brasil.fillna(0, inplace=True)
84/128: brasil
84/129: brasil.drop(['estado'])
84/130: brasil.columns
84/131: brasil.drop(['estado','municipio','coduf','codmun','codRegiaoSaude','nomeRegiaoSaude'], axis=1)
84/132: brasil
84/133: brasil.drop(['estado','municipio','coduf','codmun','codRegiaoSaude','nomeRegiaoSaude'], axis=1, inplace=True)
84/134: brasil
84/135: brasil.describe()
84/136: brasil['obitosNovos'].sum()
84/137: brasil.to_csv('brasil_10out2020.csv')
84/138: import readline
84/139: readline.write_history_file('./history10out2020')
84/140: ls
84/141: cat history10out2020
84/142: readline.write_history_file('')
84/143: %history -f ./history.py
84/144: ls
84/145: cat history.py
84/146: brasil.corr
84/147: brasil.columns
84/148: brasil[['casosAcumulado', 'obitosAcumulado']].corr
84/149: acumulados  = brasil['casosAcumulado']
84/150: acumulados
84/151: brasil[['casosAcumulado', 'obitosAcumulado']].astype(float)
84/152: brasil[['casosAcumulado', 'obitosAcumulado']].astype(int)
84/153: brasil[['casosAcumulado', 'obitosAcumulado']].astype(int)
84/154: brasil
84/155: brasil[['casosAcumulado', 'obitosAcumulado']].astype(int, inplace = True)
84/156: brasil['casosAcumulado'].astype(int)
84/157: brasil['casosAcumulado'] = brasil['casosAcumulado'].astype(int)
84/158: brasil
84/159: brasil['obitosAcumulado'] = brasil['obitosAcumulado'].astype(int)
84/160: brasil[['casosAcumulado', 'obitosAcumulado']].corr
84/161: brasil[['casosAcumulado', 'obitosAcumulado']].corr()
84/162: ?brasil[['casosAcumulado', 'obitosAcumulado']].corr()
84/163: ?.corr()
84/164: ?.corr
84/165: ?pd.DataFrame.corr
   1: import tensorflow as tf
   2: tf
   3: from tensorflow import kerar
   4: from tensorflow import keras
   5: import numpy as np
   6: import matplotlib.pyplot as plt
   7: print(tf.__version__)
   8: fashion_mnist = keras.datasets.fashion_mnist
   9: (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
  10: train_images
  11: train_labels
  12:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  13: train_images.shape
  14: len(train_labels)
  15: len(train_images)
  16: test_images.shape
  17: plt.figure()
  18: plt.imshow(train_images[0])
  19: plt.colorbar()
  20: plt.grid(False)
  21: plt.show()
  22: train_images = train_images / 255.0
  23: test_images= test_images / 255.0
  24: plt.figure(figsize=(10,10))
  25:
for i in range(25): 
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
  26: plt.show()
  27:
model = keras.Sequential([
keras.layers.Flatten(input_shape=(28,28)),
keras.layers.Dense(128, activation='relu'),
keras.layers.Dense(10, activation='softmax')
])
  28: model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  29: model.fit(train_images, train_labels, epochs = 10)
  30: test_loss, test_acc = model.evaluate(test_images, test_images, verbose=2)
  31: test_loss, test_acc = model.evaluate(test_images, test_images, verbose=2)
  32: predictions = model.predict(test_images)
  33: pre
  34: predictions
  35: predictions[0]
  36: np.argmax(predictions[0])
  37: test_images[9]
  38: test_labels[9]
  39: test_labels[0]
  40:
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  41:
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
  42:
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
  43:
# Plota o primeiro X test images, e as labels preditas, e as labels verdadeiras.
# Colore as predições corretas de azul e as incorretas de vermelho.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
  44: img = test_images[50]
  45: print(img.shape)
  46: img = (np.expand_dims(img, 0))
  47: img.shape
  48: predictions_single = model.predict(img)
  49: predictions_single
  50: np.argmax(predictions_single[0])
  51:
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
  52: plt.show()
  53: %history > tensor.py
  54: ls
  55: %history
  56: %history  -g -f tensor.py
  57: pdw
  58: pwd
  59: cd ..
  60: cd ..
  61: cd
  62: cd
  63: cd Developer/TCC/
  64: %history  -g -f tensor.py
