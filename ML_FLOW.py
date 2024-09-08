import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from itertools import combinations
from xgboost import XGBClassifier
import re
from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model, get_config
import warnings
warnings.filterwarnings("ignore")


class ML_FLOW_PARCIAL:
    def __init__(self, met, mod):
        self.met = met #selecciona si el modelo es con o sin ing. de variables
        self.mod = mod #selecciona alguno de los tres mejores modelos
        pass
        
    def load_data(self):
        ba_entr = pd.read_csv('train.csv')
        ba_pru = pd.read_csv("test.csv")
        return ba_entr, ba_pru
        
    def preprocessing(self):
        ##TRATAMIENTO DE DATOS
        category_mapping = {'Graduate': 0, 'Dropout': 1, 'Enrolled': 2}
        
        if self.met == True:
            self.ba_entr['Target'] = self.ba_entr['Target'].map(category_mapping).astype('object')
        else:
            self.ba_entr['Target'] = self.ba_entr['Target'].map(category_mapping)
            
        ct = ['Marital status', 'Application mode', 'Application order',
        'Course', 'Daytime/evening attendance', 'Previous qualification',
        'Nacionality', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Displaced', 
        'Educational special needs', 'Debtor','Tuition fees up to date', 
        'Gender', 'Scholarship holder', 'International']
        
        for k in ct:
            self.ba_entr[k] = self.ba_entr[k].astype("O")
            self.ba_pru[k] = self.ba_pru[k].astype("O")
        
        return self.ba_entr, self.ba_pru
            
    def train_model(self):
        formato = pd.DataFrame({'Variable': list(self.ba_entr.columns), 'Formato': self.ba_entr.dtypes })
        
        cuantitativas = list(formato.loc[formato["Formato"]!="object","Variable"])
        cuantitativas = [x for x in cuantitativas if x not in ["id","Target"]]
        
        if self.met == True:
            ## Variables al cuadrado
            base_cuadrado = self.ba_entr.get(cuantitativas).copy()
            base_cuadrado["Target"] = self.ba_entr["Target"].copy()

            var_names2, pvalue1 = [], []

            for k in cuantitativas:
                base_cuadrado[k+"_2"] = base_cuadrado[k] ** 2

                # Prueba de Kruskal sin logaritmo
                mue1 = base_cuadrado.loc[base_cuadrado["Target"]==0,k+"_2"].to_numpy()
                mue2 = base_cuadrado.loc[base_cuadrado["Target"]==1,k+"_2"].to_numpy()
                mue3 = base_cuadrado.loc[base_cuadrado["Target"]==2,k+"_2"].to_numpy()

                p1 = stats.kruskal(mue1,mue2,mue3)[1]

                # Guardar p values y variables
                var_names2.append(k+"_2")
                pvalue1.append(np.round(p1,2))
            pcuadrado1 = pd.DataFrame({'Variable2':var_names2,'p value':pvalue1})
            pcuadrado1["criterio"] = pcuadrado1.apply(lambda row: 1 if row["p value"]<=0.10 else 0,axis = 1)

            ## Interacciones cuantitativas
            lista_inter = list(combinations(cuantitativas,2))
            base_interacciones = self.ba_entr.get(cuantitativas).copy()
            var_interaccion, pv1 = [], []
            base_interacciones["Target"] = self.ba_entr["Target"].copy()

            for k in lista_inter:
                base_interacciones[k[0]+"__"+k[1]] = base_interacciones[k[0]] * base_interacciones[k[1]]

                # Prueba de Kruskal
                mue1 = base_interacciones.loc[base_interacciones["Target"]==0,k[0]+"__"+k[1]].to_numpy()
                mue2 = base_interacciones.loc[base_interacciones["Target"]==1,k[0]+"__"+k[1]].to_numpy()
                mue3 = base_interacciones.loc[base_interacciones["Target"]==2,k[0]+"__"+k[1]].to_numpy()
                p1 = stats.kruskal(mue1,mue2,mue3)[1]

                var_interaccion.append(k[0]+"__"+k[1])
                pv1.append(np.round(p1,2))
            pxy = pd.DataFrame({'Variable':var_interaccion,'p value':pv1})
            pxy["criterio"] = pxy.apply(lambda row: 1 if row["p value"]<=0.10 else 0, axis = 1)
            
            ## Razones
            raz1 = [(x,y) for x in cuantitativas for y in cuantitativas]
            base_razones1 = self.ba_entr.get(cuantitativas).copy()
            base_razones1["Target"] = self.ba_entr["Target"].copy()

            var_nm, pval = [], []
            for j in raz1:
                if j[0]!=j[1]:
                    base_razones1[j[0]+"__coc__"+j[1]] = base_razones1[j[0]] / (base_razones1[j[1]]+0.01)

                    # Prueba de Kruskal
                    mue1 = base_razones1.loc[base_razones1["Target"]==0,j[0]+"__coc__"+j[1]].to_numpy()
                    mue2 = base_razones1.loc[base_razones1["Target"]==1,j[0]+"__coc__"+j[1]].to_numpy()
                    mue3 = base_razones1.loc[base_razones1["Target"]==2,j[0]+"__coc__"+j[1]].to_numpy()
                    p1 = stats.kruskal(mue1,mue2,mue3)[1]
        
                    # Guardar valores
                    var_nm.append(j[0]+"__coc__"+j[1])
                    pval.append(np.round(p1,2))
            prazones = pd.DataFrame({'Variable':var_nm,'p value':pval})
            prazones["criterio"] = prazones.apply(lambda row: 1 if row["p value"]<=0.10 else 0, axis = 1)
            
            ## Interacciones categóricas
            categoricas = list(formato.loc[formato["Formato"]=="O","Variable"])
            categoricas = [x for x in categoricas if x not in ["id","Target"]]

            def nombre_(x):
              return "C"+str(x)
            cb = list(combinations(categoricas,2))
            p_value, modalidades, nombre_var = [], [], []

            base2 = self.ba_entr.get(categoricas).copy()
            for k in base2.columns:
              base2[k] = base2[k].map(nombre_)

            base2["Target"] = self.ba_entr["Target"].copy()

            for k in range(len(cb)):
                # Variable con interacción
                base2[cb[k][0]] = base2[cb[k][0]]
                base2[cb[k][1]] = base2[cb[k][1]]

                base2[cb[k][0]+"__"+cb[k][1]] = base2[cb[k][0]] + "__" + base2[cb[k][1]]

                # Prueba chi cuadrado
                c1 = pd.DataFrame(pd.crosstab(base2["Target"],base2[cb[k][0]+"__"+cb[k][1]]))
                pv = stats.chi2_contingency(c1)[1]

                # Número de modalidades por categoría
                mod_ = len(base2[cb[k][0]+"__"+cb[k][1]].unique())

                # Guardar p value y modalidades
                nombre_var.append(cb[k][0]+"__"+cb[k][1])
                modalidades.append(mod_)
                p_value.append(pv)
            pc = pd.DataFrame({'Variable':nombre_var,'Num Modalidades':modalidades,'p value':p_value})
            pc.loc[(pc["p value"]<=0.20) & (pc["Num Modalidades"]<=8),].sort_values(["p value"],ascending=True)
            ## Dummies categóricas más significativas (p value <= 0.20 y bajo número de modalidades)
            def indicadora(x):
              if x==True:
                return 1
              else:
                return 0

            seleccion1 = list(pc.loc[(pc["p value"]<=0.20) & (pc["Num Modalidades"]<=8),"Variable"])
            sel1 = base2.get(seleccion1)


            ## convertir categorica a numerica: dummies
            contador = 0
            lb1 = pd.DataFrame()
            for k in sel1:
                if contador==0:
                    lb1 = pd.get_dummies(sel1[k],drop_first=True)
                    lb1.columns = [k + "_" + x for x in lb1.columns]
                else:
                    lb2 = pd.get_dummies(sel1[k],drop_first=True)
                    lb2.columns = [k + "_" + x for x in lb2.columns]
                    lb1 = pd.concat([lb1,lb2],axis=1)
                    lb1
                contador = contador + 1
            
            for k in lb1.columns:
              lb1[k] = lb1[k].map(indicadora)

            lb1["Target"] = self.ba_entr["Target"].copy()

            ## Interacción cuantitativa vs categórica
            cat_cuanti = [(x,y) for x in cuantitativas for y in categoricas]
            v1, v2, pvalores_min, pvalores_max  = [], [], [], []

            for j in cat_cuanti:
                k1 = j[0]
                k2 = j[1]

                g1 = pd.get_dummies(self.ba_entr[k2])
                lt1 = list(g1.columns)

                for k in lt1:
                    g1[k] = g1[k] * self.ba_entr[k1]

                g1["Target"] = self.ba_entr["Target"].copy()
                pvalues_c = []
                for y in lt1:
                    mue1 = g1.loc[g1["Target"]==0,y].to_numpy()
                    mue2 = g1.loc[g1["Target"]==1,y].to_numpy()
                    mue3 = g1.loc[g1["Target"]==2,y].to_numpy()
        
                    try:
                      pval = (stats.kruskal(mue1,mue2,mue3)[1]<=0.20)

                      if pval==True:
                          pval = 1
                      else:
                          pval = 0
                    except ValueError:
                      pval = 0
                    pvalues_c.append(pval)

                min_ = np.min(pvalues_c) # Se revisa si alguna de las categorías no es significativa
                max_ = np.max(pvalues_c) # Se revisa si alguna de las categorías es significativa
                v1.append(k1) # nombre de la variable 1
                v2.append(k2) # nombre de la variable 2
                pvalores_min.append(np.round(min_,2))
                pvalores_max.append(np.round(max_,2))
            pc2 = pd.DataFrame({'Cuantitativa':v1,'Categórica':v2,'p value':pvalores_min, 'p value max':pvalores_max})
            pc2.loc[(pc2["p value"]==1) & (pc2["p value max"]==1),]

            ## Base de Feature Enginnering
            v1 = list(pc2.loc[(pc2["p value"]==1) & (pc2["p value max"]==1),"Cuantitativa"])
            v2 = list(pc2.loc[(pc2["p value"]==1) & (pc2["p value max"]==1),"Categórica"])

            for j in range(len(v1)):

                if j==0:
                    g1 = pd.get_dummies(self.ba_entr[v2[j]],drop_first=True)
                    lt1 = list(g1.columns)
                    for k in lt1:
                        g1[k] = g1[k] * self.ba_entr[v1[j]]
                    g1.columns = [v1[j] + "_" + v2[j] + "_" + str(x) for x in lt1]
                else:
                    g2 = pd.get_dummies(self.ba_entr[v2[j]],drop_first=True)
                    lt1 = list(g2.columns)
                    for k in lt1:
                        g2[k] = g2[k] * self.ba_entr[v1[j]]
                    g2.columns = [v1[j] + "_" + v2[j] + "_" + str(x) for x in lt1]
                    g1 = pd.concat([g1,g2],axis=1)

            g1["Target"] = self.ba_entr["Target"].copy()
            #### SELECCION DE VARIABLES CON XGBOOST
            var_cuad = list(pcuadrado1["Variable2"])
            base_modelo1 = base_cuadrado.get(var_cuad+["Target"])
            base_modelo1["Target"] = base_modelo1["Target"].map(int) ## convertir en enteros las categorias
            cov = list(base_modelo1.columns)
            cov = [x for x in cov if x not in ["Target"]]

            X1 = base_modelo1.get(cov)
            y1 = base_modelo1.get(["Target"])

            modelo1 = XGBClassifier()
            modelo1 = modelo1.fit(X1,y1)

            importancias = modelo1.feature_importances_
            imp1 = pd.DataFrame({'Variable':X1.columns,'Importancia':importancias})
            imp1["Importancia"] = imp1["Importancia"] * 100 / np.sum(imp1["Importancia"])
            imp1 = imp1.sort_values(["Importancia"],ascending=False)
            imp1.index = range(imp1.shape[0])

            var_int = list(pxy["Variable"])
            base_modelo2 = base_interacciones.get(var_int+["Target"])
            base_modelo2["Target"] = base_modelo2["Target"].map(int)
            cov = list(base_modelo2.columns)
            cov = [x for x in cov if x not in ["Target"]]

            X2 = base_modelo2.get(cov)
            y2 = base_modelo2.get(["Target"])

            modelo2 = XGBClassifier()
            modelo2 = modelo2.fit(X2,y2)

            importancias = modelo2.feature_importances_
            imp2 = pd.DataFrame({'Variable':X2.columns,'Importancia':importancias})
            imp2["Importancia"] = imp2["Importancia"] * 100 / np.sum(imp2["Importancia"])
            imp2 = imp2.sort_values(["Importancia"],ascending=False)
            imp2.index = range(imp2.shape[0])

            var_raz = list(prazones["Variable"])
            base_modelo3 = base_razones1.get(var_raz+["Target"])
            base_modelo3["Target"] = base_modelo3["Target"].map(int)
            cov = list(base_modelo3.columns)
            cov = [x for x in cov if x not in ["Target"]]

            X3 = base_modelo3.get(cov)
            y3 = base_modelo3.get(["Target"])

            modelo3 = XGBClassifier()
            modelo3 = modelo3.fit(X3,y3)

            importancias = modelo3.feature_importances_
            imp3 = pd.DataFrame({'Variable':X3.columns,'Importancia':importancias})
            imp3["Importancia"] = imp3["Importancia"] * 100 / np.sum(imp3["Importancia"])
            imp3 = imp3.sort_values(["Importancia"],ascending=False)
            imp3.index = range(imp3.shape[0])

            lb1["Target"] = lb1["Target"].map(int)
            cov = list(lb1.columns)
            cov = [x for x in cov if x not in ["Target"]]

            X4 = lb1.get(cov)
            y4 = lb1.get(["Target"])

            modelo4 = XGBClassifier()
            modelo4 = modelo4.fit(X4,y4)

            importancias = modelo4.feature_importances_
            imp4 = pd.DataFrame({'Variable':X4.columns,'Importancia':importancias})
            imp4["Importancia"] = imp4["Importancia"] * 100 / np.sum(imp4["Importancia"])
            imp4 = imp4.sort_values(["Importancia"],ascending=False)
            imp4.index = range(imp4.shape[0])

            g1["Target"] = g1["Target"].map(int)
            cov = list(g1.columns)
            cov = [x for x in cov if x not in ["Target"]]

            X5 = g1.get(cov)
            y5 = g1.get(["Target"])

            modelo5 = XGBClassifier()
            modelo5 = modelo5.fit(X5,y5)

            importancias = modelo5.feature_importances_
            imp5 = pd.DataFrame({'Variable':X5.columns,'Importancia':importancias})
            imp5["Importancia"] = imp5["Importancia"] * 100 / np.sum(imp5["Importancia"])
            imp5 = imp5.sort_values(["Importancia"],ascending=False)
            imp5.index = range(imp5.shape[0])

            #### VARIABLES MAS IMPORTANTES POR XGBOOST EN CADA CASO ####
            c2 = list(imp1.iloc[0:3,0]) # Variables al cuadrado
            cxy = list(imp2.iloc[0:3,0]) # Interacciones cuantitativas
            razxy = list(imp3.iloc[0:3,0]) # Razones
            catxy = list(imp4.iloc[0:3,0]) # Interacciones categóricas
            cuactxy = list(imp5.iloc[0:3,0]) # Interacción cuantitativa y categórica

            # Variables cuantitativas (Activar D1)
            D1 = self.ba_entr.get(cuantitativas).copy()

            # Variables categóricas
            D2 = self.ba_entr.get(categoricas).copy()
            for k in categoricas:
              D2[k] = D2[k].map(nombre_)
            D4 = D2.copy()

            # Variables al cuadrado (Activar D1)
            cuadrado = [re.findall(r'(.+)_\d+', item) for item in c2]
            cuadrado = [x[0] for x in cuadrado]

            for k in cuadrado:
              D1[k+"_2"] = D1[k] ** 2
            
            # Interacciones cuantitativas (Activar D1)
            result = [re.findall(r'([A-Za-z\s\(\)0-9]+)', item) for item in cxy]

            for k in result:
              D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]

            # Razones
            result2 = [re.findall(r'(.+)__coc__(.+)', item) for item in razxy]
            for k in result2:
              k2 = k[0]
              D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)

            # Interacciones categóricas
            result3 = [re.search(r'([^_]+__[^_]+)', item).group(1).split('__') for item in catxy]
            for k in result3:
              D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]

            # Interacción cuantitativa vs categórica
            D5 = self.ba_entr.copy()
            result4 = [re.search(r'(.+?)_(.+?)_1', item).groups() for item in cuactxy]
            contador = 0
            for k in result4:
              col1, col2 = k[1], k[0] # categórica, cuantitativa
              if contador == 0:
                D51 = pd.get_dummies(D5[col1],drop_first=True)
                for j in D51.columns:
                  D51[j] = D51[j] * D5[col2]
                D51.columns = [col2+"_"+col1+"_"+ str(x) for x in D51.columns]
              else:
                D52 = pd.get_dummies(D5[col1],drop_first=True)
                for j in D52.columns:
                  D52[j] = D52[j] * D5[col2]
                D52.columns = [col2+"_"+col1+"_"+ str(x) for x in D52.columns]
                D51 = pd.concat([D51,D52],axis=1)
              contador = contador + 1
            #### BASE MODELO ####
            B1 = pd.concat([D1,D4],axis=1)
            base_modelo = pd.concat([B1,D51],axis=1)
            base_modelo["Target"] = self.ba_entr["Target"].copy()
            base_modelo["Target"] = base_modelo["Target"].map(int)

            #### AUTOML ####
            formatos = pd.DataFrame(base_modelo.dtypes).reset_index()
            formatos.columns = ["Variable","Formato"]
            cuantitativas_bm = list(formatos.loc[formatos["Formato"]!="object",]["Variable"])
            categoricas_bm = list(formatos.loc[formatos["Formato"]=="object",]["Variable"])
            cuantitativas_bm = [x for x in cuantitativas_bm if x not in ["Target"]]
            categoricas_bm = [x for x in categoricas_bm if x not in ["Target"]]

            # Configuración del experimento
            exp_clf101 = setup(data=base_modelo,
            target='Target',
            session_id=123,
            train_size=0.7,
            numeric_features = cuantitativas_bm,
            categorical_features = categoricas_bm,
            fix_imbalance=False)
        else:
            formatos = pd.DataFrame(self.ba_entr.dtypes).reset_index()
            formatos.columns = ["Variable","Formato"]
            cuantitativas_bm = list(formatos.loc[formatos["Formato"]!="object",]["Variable"])
            categoricas_bm = list(formatos.loc[formatos["Formato"]=="object",]["Variable"])
            cuantitativas_bm = [x for x in cuantitativas_bm if x not in ["Target"]]
            categoricas_bm = [x for x in categoricas_bm if x not in ["Target"]]

            # Configuración del experimento
            exp_clf101 = setup(data=self.ba_entr,
            target='Target',
            session_id=123,
            train_size=0.7,
            numeric_features = cuantitativas_bm,
            categorical_features = categoricas_bm,
            fix_imbalance=False)
        
        # Comparación de modelos
        best_model = compare_models(sort='AUC') ##accuracy
        return best_model, exp_clf101, cuantitativas, categoricas, c2, cxy, razxy, catxy, cuactxy
        
    def select_model(self, exp_clf101):
        if self.mod == 1:
            dt = create_model('lightgbm')
            
            import pickle
            with open('best_model.pkl', 'wb') as model_file:
                pickle.dump(dt, model_file)
                
            #### CONTINUAR OPTIMIZACIÓN ####
            # Define the parameter grid for Grid Search
            param_grid_bayesian = {
                'n_estimators': [50,100,200],
                'max_depth': [3,5,7],
                'min_child_samples': [50,150,200]
            }
            # Perform Bayesian Search
            tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, search_library='scikit-optimize', search_algorithm='bayesian',fold=5)
        elif self.mod == 2:
            dt2 = create_model('rf')  # random forest classifier
            
            import pickle
            with open('best_model_2.pkl', 'wb') as model_file:
                pickle.dump(dt2, model_file)
                    
            #### CONTINUAR OPTIMIZACION ####
            # Define the parameter grid for Grid Search
            param_grid_bayesian_rf = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }
            # Perform Bayesian Search
            tuned_dt = tune_model(dt2, custom_grid=param_grid_bayesian_rf, search_library='scikit-optimize', search_algorithm='bayesian',fold=5)
        else:
            dt3 = create_model('et')  # extra trees classifier
            
            import pickle
            with open('best_model_3.pkl', 'wb') as model_file:
                pickle.dump(dt3, model_file)
                    
            #### CONTINUAR OPTIMIZACIÓN ####
            # Define the parameter grid for Grid Search
            param_grid_bayesian_et = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            # Perform Bayesian Search
            tuned_dt = tune_model(dt3, custom_grid=param_grid_bayesian_et, search_library='scikit-optimize', search_algorithm='bayesian',fold=5)
        # Visualización exhaustiva del modelo
        evaluate_model(tuned_dt)
                
        # Evaluar el modelo en el conjunto de prueba
        predictions_test = predict_model(tuned_dt)
    
        predictions_train = predict_model(tuned_dt, data=exp_clf101.get_config('X_train'))
    
        y_train = get_config('y_train')
        y_test = get_config('y_test')
    
        from sklearn.metrics import accuracy_score, roc_auc_score
        # Error de entrenamiento
        u1 = accuracy_score(y_train,predictions_train["prediction_label"])
        # Error de test
        u2 = accuracy_score(y_test,predictions_test["prediction_label"])
    
        # Finalizar el modelo
        final_dt = finalize_model(tuned_dt)
        return final_dt, u1, u2
        
    def predict(self, final_dt, cuantitativas, categoricas, c2, cxy, razxy, catxy, cuactxy):
        if self.met == True:
            #### PREDICCIÓN NUEVOS DATOS ####
            # Variables cuantitativas (Activar D1)
            D1 = self.ba_pru.get(cuantitativas).copy()

            # Variables categóricas
            def nombre_(x):
              return "C"+str(x)
            
            D2 = self.ba_pru.get(categoricas).copy()
            for k in categoricas:
              D2[k] = D2[k].map(nombre_)
            D4 = D2.copy()

            
            # Variables al cuadrado (Activar D1)
            cuadrado = [re.findall(r'(.+)_\d+', item) for item in c2]
            cuadrado = [x[0] for x in cuadrado]
            for k in cuadrado:
              D1[k+"_2"] = D1[k] ** 2

            # Interacciones cuantitativas (Activar D1)
            result = [re.findall(r'([A-Za-z\s\(\)0-9]+)', item) for item in cxy]
            for k in result:
              D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]

            # Razones
            result2 = [re.findall(r'(.+)__coc__(.+)', item) for item in razxy]
            for k in result2:
              k2 = k[0]
              D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)

            # Interacciones categóricas
            result3 = [re.search(r'([^_]+__[^_]+)', item).group(1).split('__') for item in catxy]
            for k in result3:
              D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]

            # Interacción cuantitativa vs categórica
            D5 = self.ba_pru.copy()
            result4 = [re.search(r'(.+?)_(.+?)_1', item).groups() for item in cuactxy]
            contador = 0
            for k in result4:
              col1, col2 = k[1], k[0] # categórica, cuantitativa
              if contador == 0:
                D51 = pd.get_dummies(D5[col1],drop_first=True)
                for j in D51.columns:
                  D51[j] = D51[j] * D5[col2]
                D51.columns = [col2+"_"+col1+"_"+ str(x) for x in D51.columns]
              else:
                D52 = pd.get_dummies(D5[col1],drop_first=True)
                for j in D52.columns:
                  D52[j] = D52[j] * D5[col2]
                D52.columns = [col2+"_"+col1+"_"+ str(x) for x in D52.columns]
                D51 = pd.concat([D51,D52],axis=1)
              contador = contador + 1

            B1 = pd.concat([D1,D4],axis=1)
            base_modelo2 = pd.concat([B1,D51],axis=1)

            df_test = base_modelo2.copy()
        else:
            df_test = self.ba_pru.copy()
        # Realizar predicciones
        predictions = predict_model(final_dt, data=df_test)

        predictions["sc"] = predictions.apply(lambda row: 1 if row["prediction_score"]<0.9 else 0, axis = 1)

        inverse_category_mapping = {0: 'Graduate', 1: 'Dropout', 2: 'Enrolled'}
        predictions['prediction_label'] = predictions['prediction_label'].map(inverse_category_mapping)
        predictions['prediction_label'] = predictions['prediction_label'].astype('category')
        
        #### ARCHIVO KAGGLE ####
        # Create a DataFrame with 'id' and 'Exited' probabilities
        result = pd.DataFrame({
            'id': self.ba_pru["id"],
            'Target': predictions['prediction_label']
        })

        # Save the result to a CSV file
        result.to_csv('modelopp.csv', index=False,sep=",")
        return result
    def evaluate_model(self, u1, u2):
        return 100 * u2
    def ML_FLOW(self):
        try:
            # Paso 1: Cargar datos
            self.ba_entr, self.ba_pru = self.load_data()
        
            # Paso 2: Preprocesamiento
            self.ba_entr, self.ba_pru = self.preprocessing()
        
            # Paso 3: Entrenamiento del modelo
            best_model, exp_clf101, cuantitativas, categoricas, c2, cxy, razxy, catxy, cuactxy = self.train_model()
         
            # Paso 4: Selección del modelo
            final_dt, u1, u2 = self.select_model(exp_clf101)
        
            # Paso 5: Predicción
            result = self.predict(final_dt, cuantitativas, categoricas, c2, cxy, razxy, catxy, cuactxy)
        
            # Paso 6: Evaluación
            metric = self.evaluate_model(u1, u2)
            return {'success':True,'accuracy':metric}
        except Exception as e:
            return {'success':False,'message':str(e)}