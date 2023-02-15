import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
# import plotly.express as px


def set_defaults():
    state.empirical_fit['Er_init'] = 1.22
    state.empirical_fit['io_init'] = 0.001
    state.empirical_fit['Ri_init'] = 0.05
    state.empirical_fit['b_init'] = 0.05
    state.empirical_fit['iL_init'] = 2.0
    state.empirical_fit['n_init'] = 0.2
    _ = state.empirical_fit.pop('fit', None)


def calc_pol_curve(i, Er, io, iL, Ri, n, b):
    V_act = b*np.log(i/io)
    V_ohm = Ri*i
    V_conc = -n*np.log(1 - i/iL)
    V = Er - V_act - V_ohm - V_conc
    return V


state = st.session_state


def main():
    if not hasattr(state, 'empirical_fit'):
        state.empirical_fit = {}
        set_defaults()

    Er_init = state.empirical_fit['Er_init']
    io_init = state.empirical_fit['io_init']
    Ri_init = state.empirical_fit['Ri_init']
    b_init = state.empirical_fit['b_init']
    iL_init = state.empirical_fit['iL_init']
    n_init = state.empirical_fit['n_init']

    Er = 1.22
    io = 0.001
    Ri = 0.05
    b = 0.05
    iL = 2.0
    n = 0.2

    st.markdown('---')
    st.latex(
        r"V_{cell} = E_r - b \cdot ln(\frac{i}{i_o}) - R_i \cdot i - n \cdot ln(1 - \frac{i}{i_L})"
    )
    st.markdown('---')

    cols = st.columns([1, 2], gap='large')

    if cols[0].button('Reset Defaults'):
        set_defaults()
        Er_init = state.empirical_fit['Er_init']
        io_init = state.empirical_fit['io_init']
        Ri_init = state.empirical_fit['Ri_init']
        b_init = state.empirical_fit['b_init']
        iL_init = state.empirical_fit['iL_init']
        n_init = state.empirical_fit['n_init']

    Er = cols[0].number_input(
        label='Standard Potential [V]',
        value=Er_init,
        min_value=0.0,
        max_value=20.0,
        step=Er_init/10,
        format="%.3f",
    )
    io = cols[0].number_input(
        label='io [A/cm^2]',
        value=io_init,
        min_value=0.0,
        max_value=100.0,
        step=io_init/10,
        format="%.7f",
    )
    b = cols[0].number_input(
        label='b - Tafel Slope',
        value=b_init,
        min_value=0.0,
        step=b_init/10,
        format="%.5f",
    )
    Ri = cols[0].number_input(
        label='Total Resistance [Ohms]',
        value=Ri_init,
        min_value=0.0,
        max_value=10.0,
        step=Ri_init/10,
        format="%.5f",
    )
    iL = cols[0].number_input(
        label='iL - Limiting Current Density [A/cm^2]',
        value=iL_init,
        min_value=0.0,
        max_value=100.0,
        step=iL_init/10,
        format="%.5f",
    )
    n = cols[0].number_input(
        label='n - Fitting Parameter 1',
        value=n_init,
        min_value=0.0,
        step=n_init/10,
        format="%.5f",
    )

    i = np.arange(io*1.01, iL, 0.005)
    V = calc_pol_curve(i, Er, io, iL, Ri, n, b)
    V_act = b*np.log(i/io)
    V_ohm = Ri*i
    V_conc = -n*np.log(1 - i/iL)

    fig, ax = plt.subplots(figsize=[5, 5])
    ax.plot(i, V, 'b-', label='Actual Cell Voltage')
    ax.plot([0, iL], [Er, Er], 'k--', alpha=0.5, label='Reversible Voltage')
    ax.plot(i, V_act, 'r--', alpha=0.5, label='Activation Overpotential')
    ax.plot(i, V_ohm, 'g--', alpha=0.5, label='Ohmic Overpotential')
    ax.plot(i, V_conc, 'm--', alpha=0.5, label='Concentration Overpotential')
    ax.set_xlim([0, iL*1.1])
    ax.set_ylim([0, Er*1.75])

    st.markdown('---')

    cols2 = st.columns([2, 1], gap='large')

    # if cols2[0].button("Use Sample Data"):
    #     from data.pol_curve import data
    #     uploaded_file = np.array(data)
    # else:
    #     uploaded_file = cols2[0].file_uploader("Choose a file")
    uploaded_file = cols2[0].file_uploader("Choose a file")
    with cols2[0].expander(label="Inspect Uploaded Data"):
        if uploaded_file is not None:
            # read csv
            try:
                df1 = pd.read_csv(uploaded_file, header=None)
            except:
                # read xls or xlsx
                df1 = pd.DataFrame(uploaded_file)
            st.dataframe(df1, use_container_width=True)
            # Add data to chart
            ax.plot(df1[0], df1[1], 'ko', alpha=0.5, label='Experimental Data')
            ax.set_xlim([0, min(iL, df1[0].max()*1.1)])

    if uploaded_file is not None:
        with cols2[1].container():
            if st.button('Fit Parameters'):
                fit = curve_fit(
                    calc_pol_curve,
                    xdata=df1[0],
                    ydata=df1[1],
                    # p0=[Er, io, iL, Ri, n, b],
                    bounds=((0, 0, 0, 0, 0, 0),(Er_init, .0010, 10, 10, 10, 10)))
                state.empirical_fit['fit'] = fit
            if 'fit' in state.empirical_fit.keys():
                fit = state.empirical_fit['fit']
                st.text(f"Er: {str(np.around(fit[0][0], decimals=8))}")
                st.text(f"io: {str(np.around(fit[0][1], decimals=8))}")
                st.text(f"iL: {str(np.around(fit[0][2], decimals=8))}")
                st.text(f"Ri: {str(np.around(fit[0][3], decimals=8))}")
                st.text(f"n: {str(np.around(fit[0][4], decimals=8))}")
                st.text(f"b: {str(np.around(fit[0][5], decimals=8))}")
                if st.button('Update Parameters'):
                    fit = state.empirical_fit['fit']
                    state.empirical_fit['Er_init'] = fit[0][0]
                    state.empirical_fit['io_init'] = fit[0][1]
                    state.empirical_fit['iL_init'] = fit[0][2]
                    state.empirical_fit['Ri_init'] = fit[0][3]
                    state.empirical_fit['n_init'] = fit[0][4]
                    state.empirical_fit['b_init'] = fit[0][5]
                    st.experimental_rerun()

    ax.set_xlabel('Current Density [$A/cm^2$]')
    ax.set_ylabel('Cell Voltage [$V$]')
    ax.legend(loc='upper left', fancybox=True, edgecolor='lightgrey')

    # Finally, render plot
    cols[1].pyplot(fig)
    # use_plotly = cols[1].radio('Plot Style', ['Matplotlib', 'Plotly'], horizontal=True, label_visibility='hidden')
    # if use_plotly == 'Plotly':
    #     cols[1].plotly_chart(fig)
    # else:
    #     cols[1].pyplot(fig)


if __name__ == "__main__":
    main()










