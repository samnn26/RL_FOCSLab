import numpy as np

def db_to_lin(x):
    return 10**(x/10)

def lin_to_db(x):
    return 10*np.log10(x)

def calculate_etaunif_nikita(n_span,gamma,l_eff,beta2,r_sym,alpha_lin,n_ch):
    '''
    calculate eta unif in [1/W^2/ch.] as defined by Nikita (agreed with Seb)
    '''
    return 1e6*(8/27)*((n_span*(gamma**2)*alpha_lin*(l_eff**2))/(np.pi*beta2*(r_sym**2)))*np.arcsinh(1e-6*(((np.pi**2)*beta2*(n_ch**2)*(r_sym**2))/(2*alpha_lin)))

def calculate_sig_ase_nikita(n_span,gain_lin,nf_lin,f_centre,r_sym):
    h_p = 6.6256*1e-34   # [J.s] Planck's constant
    return n_span*(gain_lin-1)*nf_lin*h_p*f_centre*r_sym*1e9

def calculate_throughput(r_sym, sig_ase, eta_unif):
    '''
    calculate throughput in [Tbps]
    '''
    return 1e-3*2*r_sym*np.log2(1+(1/3)*((4/(sig_ase**2*eta_unif))**(1/3)))

def calculate_max_snr(sig_ase, eta_unif):
    '''
    calculate the maximum SNR
    '''
    return (1/3)*((4/(sig_ase**2*eta_unif))**(1/3))
