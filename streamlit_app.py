import streamlit as st
import pandas as pd
import numpy as np
from scipy import interpolate
from PIL import Image
import plotly.graph_objects as go
from scipy import interpolate

st.set_page_config(page_title='คำนวณแผ่นดินไหว มยผ.1301/1302-61',layout='wide',page_icon="⚙️")

def img_show(name, caption='', width=True):
    image = Image.open(name) 
    return st.image(image, use_column_width=width, caption=caption, )

st.write('# มาตรฐานการออกแบบอาคารต้านทานการสั่นสะเทือนของแผ่นดินไหว')
st.write('# มยผ.1301/1302-61')

st.write('### ค่าตัวประกอบความสำคัญและประเภทของอาคาร')
col1, col2 = st.columns(2)
with col1:
    col1, buff = st.columns([0.5,0.5])
    with col1:
        img_show('EQ_IF01.png')
important_dict = {
    'น้อย': 1.0,
    'ปกติ': 1.0,
    'มาก': 1.25,
    'สูงมาก': 1.5,
}
important = st.selectbox(label='ประเภทความสำคัญ', options=important_dict.keys(), key='important')
I = important_dict[important]
st.write(r'Important factor, $I = %.2f$'%(I))


st.write('### วิธีการวิเคราะห์โครงสร้างเพื่อคำนวณผลของแรงแผ่นดินไหว')
cal_method = ['วิธีสถิตเทียบเท่า', 'วิธีเชิงพลศาสตร์']
cal = st.radio(label='วิธีการวิเคราะห์', options=cal_method, index=0, key='cal',horizontal=True)


st.write('### รายละเอียดโครงสร้าง')
col1, col2, col3  = st.columns(3)
with col1:
    structure_list = ['คอนกรีตเสริมเหล็ก', 'เหล็ก']
    structure = st.radio(label='ประเภทโครงสร้าง', options=structure_list, index=0, key='structure')
with col2:
    if structure == structure_list[0]:
        damping_list = ['5.0%', '2.5%']
    else:
        damping_list = ['2.5%']
    damping = st.radio(label='ความหน่วง', options=damping_list, index=0, key='damping')
with col3:
    H = st.number_input(label=r'ความสูงอาคารวัดจากพื้นดิน, $H \mathrm{~[m]}$', min_value=0.0, value=6.0, key='H')


st.write('### ความเร่งตอบสนองเชิงสเปกตรัม')

bkk = st.checkbox('พื้นที่ในแอ่งกรุงเทพฯ หรือไม่ ???', value=False, key='bkk')

if not bkk:

    df_SsS1 = pd.read_excel('แผ่นดินไหว_table.xlsx', sheet_name='SsS1')

    col1, col2, col3 =st.columns(3)
    with col1:
        province = st.selectbox(label='จังหวัด', options=df_SsS1['จังหวัด'].unique(), index=12, key='province')
    with col2:
        district = st.selectbox(label='อำเภอ', options=df_SsS1.loc[df_SsS1['จังหวัด']==province,'อำเภอ'], index=8, key='district')

    Ss = df_SsS1.loc[(df_SsS1['จังหวัด']==province) & (df_SsS1['อำเภอ']==district),'Ss'].iloc[0]
    S1 = df_SsS1.loc[(df_SsS1['จังหวัด']==province) & (df_SsS1['อำเภอ']==district),'S1'].iloc[0]

    st.write(r'$S_s = %.3f \> g$'%(Ss))
    st.write(r'$S_1 = %.3f \> g$'%(S1))
    
    st.write('### ปรับแก้ค่าระดับความรุนแรงจากแรงแผ่นดินไหวเนื่องจากผลของประเภทชั้นดิน')

    soil_t = st.selectbox(label='ประเภทชั้นดิน', options=['A','B','C','D','E','F'], index=0, key='soil_t')

    def FaFv(df,S):
        if S <= df['index'].min():
            F = df[soil_t].iloc[0]
        elif S >= df['index'].max():
            F = df[soil_t].iloc[-1]
        else:
            y0 = df.loc[df['index'] <= S,soil_t].iloc[-1]
            y1 = df.loc[df['index'] >= S,soil_t].iloc[0]

            x0 = df.loc[df['index'] <= S,'index'].iloc[-1]
            x1 = df.loc[df['index'] >= S,'index'].iloc[0]

            x_data = [x0, x1]
            y_data = [y0, y1]

            f = interpolate.interp1d(x_data, y_data)

            F = f([S])[0]
        
        return F

    df_Fa = pd.read_excel('แผ่นดินไหว_table.xlsx', sheet_name='Fa')
    df_Fa.set_index('ประเภทชั้นดิน', inplace=True)
    df_Fa = df_Fa.T.reset_index().astype('float')

    df_Fv = pd.read_excel('แผ่นดินไหว_table.xlsx', sheet_name='Fv')
    df_Fv.set_index('ประเภทชั้นดิน', inplace=True)
    df_Fv = df_Fv.T.reset_index().astype('float')

    Fa = FaFv(df_Fa,Ss)
    Fv = FaFv(df_Fv,S1)

    st.write(r'$F_a = %.3f \> g$'%(Fa))
    st.write(r'$F_v = %.3f \> g$'%(Fv))

    SMS = Fa*Ss
    SM1 = Fv*S1

    st.write(r'$S_{MS} = F_{a} S_{s} = %.3f \times %.3f = %.3f \> g$'%(Fa,Ss,SMS))
    st.write(r'$S_{M1} = F_{v} S_{1} = %.3f \times %.3f = %.3f \> g$'%(Fv,S1,SM1))

    st.write('### ปรับแก้ค่าระดับความรุนแรงจากแรงแผ่นดินไหวสำหรับออกแบบ')

    SDS = (2/3)*SMS
    SD1 = (2/3)*SM1

    st.write(r'$S_{DS} = \frac{2}{3} S_{MS} = \frac{2}{3} \times %.3f = %.3f \> g$'%(SMS,SDS))
    st.write(r'$S_{D1} = \frac{2}{3} S_{M1} = \frac{2}{3} \times %.3f = %.3f \> g$'%(SM1,SD1))
    
    
    
else:
    with (st.expander('การแบ่งโซนพื้นที่ในแอ่งกรุงเทพฯ')):
        img_show('eq_bkk_zone.png')
    
    zone = st.selectbox(label='Zone', options=np.arange(1,11), key='zone')
    
    if cal == cal_method[0]:
        sheet_name_ = 'bkk_equivalent'
    else:
        sheet_name_ = 'bkk_rsa'
        
    if damping == '5.0%':
        sheet_name = sheet_name_ + '_5.0'
    else:
        sheet_name = sheet_name_ + '_2.5'
    
    df_bkk = pd.read_excel('แผ่นดินไหว_table.xlsx', sheet_name=sheet_name)
    
    col = df_bkk.columns
    df_bkk = pd.melt(df_bkk, id_vars=col[0], value_vars=col[1:],var_name='T', value_name='Sa').astype('float')
    
    SDS = df_bkk.loc[(df_bkk['zone']==zone) & (df_bkk['T']==0.2),'Sa'].iloc[0]
    SD1 = df_bkk.loc[(df_bkk['zone']==zone) & (df_bkk['T']==1.0),'Sa'].iloc[0]
    
    st.write(r'$S_{DS} = %.3f \> g$'%(SDS))
    st.write(r'$S_{D1} = %.3f \> g$'%(SD1))


st.write('### คำนวณค่าคาบการสั่นพื้นฐานโดยประมาณ')
#หาค่า T
if structure == structure_list[0]:
    T_val = 0.02*H
    st.write('อาคาร',structure,r'$\qquad  T = 0.02H = 0.02 \times %.2f \mathrm{~m} = %.3f \mathrm{~sec}$'%(H,T_val))
else:
    T_val = 0.03*H
    st.write('อาคาร',structure,r'$\qquad  T = 0.03H = 0.03 \times %.2f \mathrm{~m} = %.3f \mathrm{~sec}$'%(H,T_val))

    
st.write('### ประเภทการออกแบบต้านทานแผ่นดินไหว')
st.write('การแบ่งประเภทการออกแบบต้านทานแผ่นดินไหวโดยพิจารณาจากค่า $S_{DS}$ และ $S_{D1}$ นี้ กำหนดให้พิจารณาอัตราส่วนความหน่วงเท่ากับร้อยละ 5 **สำหรับอาคารทุกประเภท**')
   
type_dict = {
    '1': 'ก',
    '2': 'ข',
    '3': 'ค',
    '4': 'ง',
}

def type161162(SDS, SD1):
    df = pd.read_excel('แผ่นดินไหว_table.xlsx', sheet_name='T1.6-1')
    df = pd.melt(df, id_vars=['min','max'], value_vars=['น้อย','ปกติ','มาก','สูงมาก'], var_name='important', value_name='type')
    type161 = df.loc[(df['min']<=SDS) & (df['max']>SDS) & (df['important']==important), 'type'].iloc[0]

    df = pd.read_excel('แผ่นดินไหว_table.xlsx', sheet_name='T1.6-2')
    df = pd.melt(df, id_vars=['min','max'], value_vars=['น้อย','ปกติ','มาก','สูงมาก'], var_name='important', value_name='type')
    type162 = df.loc[(df['min']<=SD1) & (df['max']>SD1) & (df['important']==important), 'type'].iloc[0]
    
    if SD1 <= SDS:
        TS = SD1/SDS
    else:
        TS = 1.0
    
    return type161, type162, TS


if bkk and damping=='2.5%':
    
    df_bkkx = pd.read_excel('แผ่นดินไหว_table.xlsx', sheet_name=sheet_name_ + '_5.0')
    
    colx = df_bkkx.columns
    df_bkkx = pd.melt(df_bkkx, id_vars=col[0], value_vars=col[1:],var_name='T', value_name='Sa').astype('float')
    
    SDSx = df_bkkx.loc[(df_bkkx['zone']==zone) & (df_bkkx['T']==0.2),'Sa'].iloc[0]
    SD1x = df_bkkx.loc[(df_bkkx['zone']==zone) & (df_bkkx['T']==1.0),'Sa'].iloc[0]
    
    st.write('สำหรับอัตราส่วนความหน่วงเท่ากับ 5%')
    st.write(r'$S_{DS} = %.3f \> g$'%(SDSx))
    st.write(r'$S_{D1} = %.3f \> g$'%(SD1x))
    
    type161, type162, TTSS = type161162(SDSx, SD1x)
    
else:
    type161, type162, TTSS = type161162(SDS, SD1)
    
if not bkk:
    if T_val < 0.8*TTSS:
        st.write(r'สำหรับ $\quad T = %.3f \mathrm{~sec} \quad < \quad 0.8 T_s = 0.8 \times %.3f = %.3f \mathrm{~sec}$'%(T_val, TTSS, 0.8*TTSS))
        st.write(r'พิจารณาประเภทการออกแบบต้านทานแผ่นดินไหวตามเกณฑ์ในตารางที่ <span style="color:red">**1.6-1**</span> เท่านั้น',unsafe_allow_html=True)
        type_num = type161
    else:
        st.write(r'สำหรับ $\quad T = %.3f \mathrm{~sec} \quad \ge \quad 0.8 T_s = 0.8 \times %.3f = %.3f \mathrm{~sec}$'%(T_val, TTSS, 0.8*TTSS))
        st.write(r'พิจารณาประเภทการออกแบบต้านทานแผ่นดินไหวที่เข้มงวดกว่าตามเกณฑ์ในตารางที่ <span style="color:red">**1.6-1 และ 1.6-2**</span>',unsafe_allow_html=True)
        type_num = max(type161,type162)
else:
    if T_val <= 0.5:
        st.write(r'สำหรับ $\quad T = %.3f \mathrm{~sec} \quad \le \quad 0.5 \mathrm{~sec}$'%(T_val))
        st.write(r'พิจารณาประเภทการออกแบบต้านทานแผ่นดินไหวตามเกณฑ์ในตารางที่ <span style="color:red">**1.6-1**</span> เท่านั้น',unsafe_allow_html=True)
        type_num = type161
    else:
        st.write(r'สำหรับ $\quad T = %.3f \mathrm{~sec} \quad > \quad 0.5 \mathrm{~sec}$'%(T_val))
        st.write(r'พิจารณาประเภทการออกแบบต้านทานแผ่นดินไหวตามเกณฑ์ในตารางที่ <span style="color:red">**1.6-2**</span> เท่านั้น',unsafe_allow_html=True)
        type_num = type162

type = type_dict[str(type_num)]
col1, col2 = st.columns(2)
with col1:
    col1, buff = st.columns([0.5,0.5])
    with col1:
        img_show('161162.png')
st.write(r'ประเภทการออกแบบต้านแผ่นดินไหว: <span style="color:red">**ประเภท %s**</span>'%(type),unsafe_allow_html=True)


st.write('### ค่าประกอบตามประเภทโครงสร้าง')
col1x, col2x, col3x = st.columns(3)
with col1x:
    col1x, buff = st.columns([0.5,0.5])
    with col1x:
        img_show('EQ_typef01.png')
        img_show('EQ_typef04.png')
        img_show('EQ_typef05.png')
with col2x:
    col2x, buff = st.columns([0.5,0.5])
    with col2x:
        img_show('EQ_typef02.png') 
with col3x:
    col3x, buff = st.columns([0.5,0.5])
    with col3x:
        img_show('EQ_typef03.png')
        
col1, col2, col3 = st.columns(3)
with col1:
    R = st.number_input(label='ค่าตัวประกอบปรับผลตอบสนอง(Response Modification Factor), $R$', min_value=0.0, value=8.0, key='R')
with col2:
    omega0 = st.number_input(label='ค่าตัวประกอบกำลังส่วนเกิน(System Overstrength Factor), $\Omega_0$', min_value=0.0, value=3.0, key='omega0')
with col3:
    Cd = st.number_input(label='ค่าตัวประกอบขยายค่าการโก่งตัว(Deflection Amplification Factor), $C_d$', min_value=0.0, value=5.5, key='Cd')



st.write('### ค่าความเร่งตอบสนองเชิงสเปกตรัมสำหรับออกแบบ')
#กราฟหาความเร่งตอบสนองเชิงสเปกตรัม Sa
if not bkk:
    if cal == cal_method[0]:
        if SD1 <= SDS:
            T0 = 0.0
            Ts = SD1/SDS
            T_data = np.append([T0,Ts],np.arange(round(Ts,1),2.1,0.1))
            S_data = np.array([SDS,SDS])

            for T in T_data:
                if T > Ts:
                    S_data = np.append(S_data,[SD1/T])
            
            # คำนวณหาค่า Sa
            if T_val <= Ts:
                Sa_val = SDS
            else:
                Sa_val = SD1/T_val
                    
        elif SD1 > SDS:
            T0 = 0.2
            Ts = 1.0
            T_data = np.append([0,T0,Ts],np.arange(1.1,2.1,0.1))
            S_data = np.array([SDS,SDS,SD1])

            for T in T_data:
                if T > Ts:
                    S_data = np.append(S_data,[SD1/T])
            
            # คำนวณหาค่า Sa
            if T_val <= T0:
                Sa_val = SDS
            elif T_val > T0 and T_val <= Ts:
                f = interpolate.interp1d([T0,Ts], [SDS,SD1])
                Sa_val = f(T_val)
            else:
                Sa_val = SD1/T_val

    elif cal == cal_method[1]:
        if SD1 <= SDS:
            T0 = 0.2*SD1/SDS
            Ts = SD1/SDS
            T_data = np.append([0,T0,Ts],np.arange(round(Ts,1),2.1,0.1))
            S_data = np.array([0.4*SDS,SDS,SDS])

            for T in T_data:
                if T > Ts:
                    S_data = np.append(S_data,[SD1/T])
            
            # คำนวณหาค่า Sa
            if T_val <= T0:
                f = interpolate.interp1d([0.0,T0], [0.4*SDS,SDS])
                Sa_val = f(T_val)
            elif T_val > T0 and T_val <= Ts:
                Sa_val = SDS
            else:
                Sa_val = SD1/T_val
                
        elif SD1 > SDS:
            T0 = 0.2
            Ts = 1.0
            T_data = np.append([0,T0,Ts],np.arange(1.1,2.1,0.1))
            S_data = np.array([0.4*SDS,SDS,SD1])

            for T in T_data:
                if T > Ts:
                    S_data = np.append(S_data,[SD1/T])
            
            # คำนวณหาค่า Sa
            if T_val <= T0:
                f = interpolate.interp1d([0.0,T0], [0.4*SDS,SDS])
                Sa_val = f(T_val)
            elif T_val > T0 and T_val <= Ts:
                f = interpolate.interp1d([T0,Ts], [SDS,SD1])
                Sa_val = f(T_val)
            else:
                Sa_val = SD1/T_val

    #ความหน่วง 2.5%
    if damping == '2.5%':
        for i in range(len(T_data)):
            if T_data[i] >= T0:
                S_data[i] = S_data[i]/0.85
            else:
                S_data[i] = SDS*(3.88*T_data[i]/Ts + 0.4)
                
        if T_val >= T0:
            Sa_val = Sa_val/0.85
        else:
            Sa_val = SDS*(3.88*T_val/Ts + 0.4)
    
elif bkk:
    
    T_data = df_bkk.loc[df_bkk['zone']==zone, 'T']
    S_data = df_bkk.loc[df_bkk['zone']==zone, 'Sa']    
    
    y0 = df_bkk.loc[(df_bkk['zone']==zone) & (df_bkk['T']<=T_val), :].iloc[-1]['Sa']
    y1 = df_bkk.loc[(df_bkk['zone']==zone) & (df_bkk['T']>=T_val), :].iloc[0]['Sa']
    x0 = df_bkk.loc[(df_bkk['zone']==zone) & (df_bkk['T']<=T_val), :].iloc[-1]['T']
    x1 = df_bkk.loc[(df_bkk['zone']==zone) & (df_bkk['T']>=T_val), :].iloc[0]['T']
    
    
    
    #log interpolation
    f = interpolate.interp1d([np.log10(x0),np.log10(x1)], [np.log10(y0),np.log10(y1)])
    Sa_val = 10**f([np.log10(T_val)])[0]
    
    
#plot function
def response_spectrum_plot(T_data,S_data):
    fig = go.Figure()

    #กราฟ
    fig.add_trace(
        go.Scatter(
            x = T_data,
            y = S_data,
            mode = 'lines+markers',
            line = dict(color='purple', width=2,),
            showlegend = False,
            hoverinfo = 'skip',
        )
    )
    
    #เส้นแนวนอน
    fig.add_trace(
        go.Scatter(
            x = [min(T_data),T_val],
            y = [Sa_val,Sa_val],
            mode = 'lines', #'lines+text'
            line = dict(dash='dash', width=3,color='red'),
            # text=[r'%.3f'%(Sa_val), ''],
            # textfont=dict(color='red', size=16),
            # textposition="top right",
            hoverinfo = 'skip',
            showlegend = False,
        )
    )
    if bkk:
        x = np.log10(min(T_data))
        y = np.log10(Sa_val)
    else:
        x = min(T_data)
        y = Sa_val
    fig.add_annotation(
            x=x, y=y,
            text=r'%.3f'%(Sa_val),
            xanchor="left",
            yanchor="bottom",
            font=dict(
                color="red",
                size=16
            ),
            showarrow=False,
            
    )
    
    #เส้นแนวตั้ง
    fig.add_trace(
        go.Scatter(
            x = [T_val,T_val],
            y = [0.0,Sa_val],
            mode = 'lines', #'lines+text'
            line = dict(dash='dash', width=3,color='red'),
            # text=[r'%.3f'%(T_val), ''],
            # textfont=dict(color='red', size=16),
            # textposition="top right",
            hoverinfo = 'skip',
            showlegend = False,
        )
    )
    if bkk:
        x = np.log10(T_val)
        y = np.log10(0.01)
    else:
        x = T_val
        y = 0.0
    fig.add_annotation(
            x=x, y=y,
            text=r'%.3f'%(T_val),
            xanchor="left",
            yanchor="bottom",
            font=dict(
                color="red",
                size=16
            ),
            showarrow=False,
            
    )    
    
    #มาร์คจุด
    fig.add_trace(
        go.Scatter(
            x = [T_val],
            y = [Sa_val],
            mode = 'markers',
            marker = dict(color='red', size=8),
            showlegend = False,
            hoverinfo = 'skip',
        )
    )
    
    fig.update_layout(
        xaxis = dict(
                        title = 'T (second)',
                        fixedrange=True,
                        range=[0.0, 2.0],
                        rangemode = "nonnegative",
                        # dtick = 0.25,
                    ),
        yaxis = dict(
                        title = 'Sa (g)',
                        fixedrange=True,
                        range=[0.0,max(S_data)+0.05],
                        scaleanchor = "x", scaleratio = 1,
                        rangemode = "nonnegative",
                        # dtick = 0.2,
                    ),
        margin = dict(t=20, b=40),
        height=300,
    )
    
    if bkk:
        fig.update_xaxes(range=[np.log10(0.01),np.log10(10)],type="log")
        fig.update_yaxes(range=[np.log10(0.01),np.log10(1)],type="log")

    return fig

col1, col2 = st.columns([0.7,0.3])
with col1:
    st.plotly_chart(response_spectrum_plot(T_data,S_data),theme=None, use_container_width=True)
    
    if structure == structure_list[0]:
        st.write(r'Period of structure, $T = %.3f \mathrm{~sec}$'%(T_val))
    else:
        st.write(r'Period of structure, $T = %.3f \mathrm{~sec}$'%(T_val))
        
    st.write(r'Acceleration of structure, $S_a = %.3f \mathrm{~g}$'%(Sa_val))
    
with col2:
    df = pd.DataFrame({
        'T (second)': T_data,
        'Sa (g)': S_data
    })
    
    

    st.dataframe(df,hide_index=True, use_container_width=True)



st.write('### สำหรับวิธีแรงสถิตเทียบเท่า (Equivalent Static)')

W = st.number_input(label='น้ำหนักโครงสร้างประสิทธิผลของอาคาร, $W \mathrm{~[tonne]}$', min_value=0.0, value=500.0)

st.write('**สัมประสิทธิ์ผลตอบสนองแรงแผ่นดินไหว, $C_s$**')
Cs_ = Sa_val*I/R
Cs = max(Cs_,0.01)

st.markdown(r'$C_s = S_a \left( \frac{I}{R} \right) \qquad\qquad \ge \qquad 0.01$')
st.markdown(r'$\quad\>\> = %.3f \left( \frac{%.2f}{%.2f} \right) \qquad\>\> \ge \qquad 0.01$'%(Sa_val,I,R))
st.markdown(r'$\quad\>\> = %.3f \qquad\qquad\quad \ge \qquad 0.01$'%(Cs_))
st.markdown(r'$\quad\>\> = %.3f$'%(Cs))

st.write('**แรงเฉือนที่ฐานอาคาร, $V$**')
V = Cs*W
st.markdown(r'$V = C_s W$')
st.markdown(r'$\quad = %.3f \mathrm{~g} \times %.2f \mathrm{~tonne}$'%(Cs,W))
st.markdown(r'$\quad = %.2f \mathrm{~tonne}$'%(V))


st.write('### การกระจายแรงเฉือนที่ฐานเป็นแรงกระทำด้านข้าง')
df_v_distribute = pd.DataFrame({
    'Floor': [4,3,2,1],
    'Wi [tonne]': [125.0,125.0,125.0,125.0],
    'Floor height [m]': [3.5,3.5,3.5,3.5],
})

st.write('**ค่าสัมประสิทธิ์กำหนดรูปแบบการกระจายแรง**')
if T_val <= 0.5:
    k = 1.0
    st.write(r'สำหรับ $\qquad T \le 0.5 \mathrm{~sec}, \qquad k = 1.0$')
elif T_val >= 2.5:
    k = 2.0
    st.write(r'สำหรับ $\qquad T \ge 2.5 \mathrm{~sec}, \qquad k = 2.0$')
else:
    k = 1 + (T_val-0.5)/2
    st.write(r'สำหรับ $\qquad 0.5 \mathrm{~sec} < T < 2.5 \mathrm{~sec}, \qquad k = 1+ \frac{T-0.5}{2} = 1+ \frac{%.2f-0.5}{2} = %.2f $'%(T_val,k))

st.write('**ตัวประกอบการกระจายแนวดิ่ง**')
st.write(r'$C_{v x}=\frac{w_x h_x^k}{\sum_{i=1}^{n} w_i h_i^k}$')

st.write(r'$F_x = C_{v x} V$')

st.write(r'โดยที่ $C_{v x}$ คือ ตัวประกอบการกระจายในแนวดิ่ง')
st.write(r'$w_i$ และ $w_x$ คือ น้ำหนักโครงสร้างประสิทธิผล')
st.write(r'$h_i$  และ  $h_x$ คือ ความสูงที่ระดับชั้น i และ x ตามลำดับ')
st.write(r'$k$ คือ ค่าสัมประสิทธิ์ที่กำหนดรูปแบบการกระจายแรง')

col1, col2 = st.columns(2)
with col1:
    col1, buff = st.columns([0.5,0.5])
    with col1:
        img_show('def_calF.png')


col1, col2 = st.columns(2)
with col1:
    st.write('**ระบุข้อมูลตามตารางดังต่อไปนี้** (สามารถแก้ไขตารางได้)')
    df_v_distribute = st.data_editor(df_v_distribute, num_rows="dynamic", key='df_v_distribute')
with col2:
    st.write('**ผลลัพธ์**')
    df_v_cal = pd.DataFrame()
    df_v_cal[r'hi [m]'] = df_v_distribute.loc[::-1, 'Floor height [m]'].cumsum()[::-1]
    wihik = df_v_distribute['Wi [tonne]']*(df_v_cal[r'hi [m]']**k)
    df_v_cal['Cvx'] = wihik / wihik.sum()
    df_v_cal['Fx [tonne]'] = df_v_cal['Cvx']*V
    df_v_cal['Vi [tonne]'] = df_v_cal['Fx [tonne]'].cumsum()
        
    st.dataframe(df_v_cal, hide_index=True)
