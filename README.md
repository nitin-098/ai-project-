# ai-project-
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title='Smart Spend | Expense Predictor', page_icon='💸', layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1250px}h1{color:#126e65}div[data-testid="stMetric"]{background:#f2f8f6;padding:18px;border-radius:14px;border:1px solid #d9eae6}</style>''', unsafe_allow_html=True)
CATEGORIES=['housing','groceries','transport','utilities','healthcare','entertainment','shopping','dining_out','education']
@st.cache_data
def load_data():
    return pd.read_csv('smart_spend_dataset.csv')
@st.cache_resource
def train_model():
    df=load_data().dropna(subset=['monthly_expense'])
    X=df.drop(columns=['monthly_expense']); y=df['monthly_expense']
    cat=['city_tier','employment_type']
    pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat)],remainder='passthrough')
    model=Pipeline([('pre',pre),('model',RandomForestRegressor(n_estimators=180,random_state=42,min_samples_leaf=2,n_jobs=-1))])
    model.fit(X,y)
    return model

def money(v):return f'₹{v:,.0f}'

df=load_data()
st.title('💸 Smart Spend')
st.caption('AI-powered monthly expense estimation and budget insights')
with st.sidebar:
    st.header('Your financial profile')
    income=st.number_input('Monthly income (₹)',min_value=0,max_value=10000000,value=60000,step=1000)
    dependents=st.number_input('Number of dependents',min_value=0,max_value=20,value=1)
    city=st.selectbox('City tier',sorted(df.city_tier.dropna().unique().tolist()))
    employment=st.selectbox('Employment type',sorted(df.employment_type.dropna().unique().tolist()))
    st.divider()
    st.header('Monthly spending (₹)')
    defaults={'housing':15000,'groceries':5000,'transport':3000,'utilities':2500,'healthcare':1500,'entertainment':2000,'shopping':2500,'dining_out':2000,'education':2000}
    expenses={c:st.number_input(c.replace('_',' ').title(),min_value=0.0,max_value=10000000.0,value=float(defaults[c]),step=500.0) for c in CATEGORIES}

features={'monthly_income':income,**expenses,'dependents':dependents,'city_tier':city,'employment_type':employment}
X=pd.DataFrame([features])[df.drop(columns=['monthly_expense']).columns]
prediction=max(0.0,float(train_model().predict(X)[0]))
entered=sum(expenses.values())
savings=income-prediction
c1,c2,c3,c4=st.columns(4)
c1.metric('Monthly income',money(income))
c2.metric('Predicted expenses',money(prediction))
c3.metric('Estimated savings',money(savings))
c4.metric('Entered expenses',money(entered))
st.info('Prediction is an estimate from a model trained on the provided dataset, not a guaranteed future expense. Entered expenses are the sum of your category inputs.')
left,right=st.columns([1.1,1])
with left:
    st.subheader('Spending by category')
    chart_df=pd.DataFrame({'Category':[c.replace('_',' ').title() for c in CATEGORIES],'Amount':list(expenses.values())})
    chart_df=chart_df[chart_df.Amount>0]
    if len(chart_df):
        fig=px.pie(chart_df,values='Amount',names='Category',hole=.55,color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),legend=dict(orientation='h',y=-.15))
        st.plotly_chart(fig,use_container_width=True)
    else:st.write('Enter spending amounts to see the breakdown.')
with right:
    st.subheader('Budget overview')
    compare=pd.DataFrame({'Metric':['Income','Predicted expenses','Entered expenses'],'Amount':[income,prediction,entered]})
    fig=px.bar(compare,x='Metric',y='Amount',color='Metric',color_discrete_sequence=['#178d80','#ef9a52','#6388c5'])
    fig.update_layout(showlegend=False,margin=dict(l=10,r=10,t=10,b=10),yaxis_title='Amount (₹)',xaxis_title='')
    st.plotly_chart(fig,use_container_width=True)
    if savings>=0:st.success(f'Your estimated savings rate is {savings/income:.1%}.' if income else 'No income entered.')
    else:st.warning('Predicted expenses exceed your entered income. Review your budget.')

st.subheader('Expense details')
st.dataframe(pd.DataFrame({'Category':[c.replace('_',' ').title() for c in CATEGORIES],'Monthly spending (₹)':list(expenses.values())}),hide_index=True,use_container_width=True)
result=pd.DataFrame([{**features,'predicted_monthly_expense':round(prediction,2),'estimated_savings':round(savings,2)}])
st.download_button('⬇️ Download prediction as CSV',data=result.to_csv(index=False),file_name='smart_spend_prediction.csv',mime='text/csv')
with st.expander('About the model'):
    st.write('A Random Forest regression model is trained on the supplied 600-row CSV using all input fields to predict monthly_expense. The model is trained locally on app startup and cached. No accuracy metric is claimed without held-out evaluation.')
