"""TAISIA 0.5 Windows GUI: responsive tabs, pure analytic tomographic audio synthesis.
Requires Tkinter, NumPy, SciPy and Matplotlib. Optional sounddevice for playback.
"""
from __future__ import annotations
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import traceback
import numpy as np
from scipy.integrate import trapezoid
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from taisia.physics import KINDS, State, wave, wave_from_coeff, tomogram, phase_fix, basis
from taisia.reconstruction import recover
from taisia.direct import recover_direct
from scipy.interpolate import CubicSpline
from taisia.forward import optical_from_wave
from taisia.audio import render, save_wav, AUDIO_MODES, MODE_IQ_AUDIBLE, MODE_IQ_RAW

try:
    import sounddevice as sd
except (ImportError,OSError):
    sd=None

METHODS=('Прямое: w → s (основной)', 'Из томограмм (Фок)', 'Аналитическая ψ (эталон)')
MODES=AUDIO_MODES
TIMING_MANUAL='Ручная настройка'
TIMING_PROFILES={
    'Короткие, но более выразительные': ('0.015','0.65'),
    'Протяжённые пакеты — рекомендую': ('0.025','1.0'),
    'Медленные, растянутые звуки': ('0.04','1.5'),
}
# Single state, with no looping. Duration is part of the preset.
SINGLE_PULSE_PROFILES={
    'Очень медленный · один импульс': ('0.35','10.0'),
    'Предельно медленный · один импульс': ('0.70','20.0'),
}

PRESETS={
    'Керр: фазовые переливы':{'kind':KINDS[10],'q0':'3.0','p0':'1.0','kerr':'1.1','time_scale':'.007','carrier_hz':'330','pulse_period':'.43'},
    'Компас: четыре компоненты':{'kind':KINDS[7],'q0':'2.7','p0':'0.8','time_scale':'.007','carrier_hz':'520','pulse_period':'.45'},
    'Фотон добавлен: m=3':{'kind':KINDS[9],'q0':'1.8','p0':'1.5','m':'3','time_scale':'.006','carrier_hz':'680','pulse_period':'.38'},
    'q-деформация: κ=0.8':{'kind':KINDS[11],'q0':'3.4','p0':'1.2','deformation':'.8','time_scale':'.007','carrier_hz':'390','pulse_period':'.41'},
    'Кот: двойной удар':{'kind':KINDS[4],'q0':'3.2','p0':'0.0','time_scale':'.012','carrier_hz':'430','pulse_period':'.55'},
    'Юрке — Столер: интерференция':{'kind':KINDS[8],'q0':'2.8','p0':'1.1','time_scale':'.009','carrier_hz':'440','pulse_period':'.42'},
    'Фоковское |8>: узлы':{'kind':KINDS[1],'n':'8','time_scale':'.009','carrier_hz':'560','pulse_period':'.48'},
    'Суперпозиция |0>+i|11>':{'kind':KINDS[6],'n0':'0','n1':'11','ratio':'0.95','phase':'1.5707963267948966','time_scale':'.009','carrier_hz':'500','pulse_period':'.48'},
    'Вакуум: контроль':{'kind':KINDS[0],'time_scale':'.012','carrier_hz':'660','pulse_period':'.45'},
    'Сжатое |6>: частотный чирп':{'kind':KINDS[12],'n':'6','a':'0.75','b':'0.9','q0':'0.0','p0':'0.0','time_scale':'.015','carrier_hz':'410','pulse_period':'.65'},
    'Биномиальное N=10, p=0.5':{'kind':KINDS[13],'n':'10','binomial_p':'0.5','phase':'1.15','time_scale':'.018','carrier_hz':'380','pulse_period':'.7'},
    'Смещённое |5>: q₀=1.5, p₀=2':{'kind':KINDS[14],'n':'5','q0':'1.5','p0':'2.0','time_scale':'.013','carrier_hz':'440','pulse_period':'.65'},
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('TAISIA 0.5 — Direct Tomogram-to-Signal')
        sw,sh=self.winfo_screenwidth(),self.winfo_screenheight()
        width=max(700,min(1200,sw-65))
        height=max(480,min(770,sh-70))
        self.geometry(f'{width}x{height}+20+20')
        self.minsize(min(800,max(500,sw-30)),min(550,max(420,sh-30)))
        self._style()
        defaults={
            'kind':KINDS[10],'n':'1','n0':'0','n1':'1','ratio':'0.8','phase':'1.0',
            'q0':'3.0','p0':'1.0','a':'0.7','b':'0.55',
            'm':'1','kerr':'1.1','deformation':'0.45','binomial_p':'0.5',
            'duration':'4.0','time_scale':'0.007','carrier_hz':'330','gain':'0.6',
            'pulse_period':'0.43','repeat_pulses':'1',
            'timing_profile':TIMING_MANUAL,
            'preset':'Керр: фазовые переливы',
            'dimension':'16','theta':'45', 'method':METHODS[0], 'mode':MODE_IQ_AUDIBLE
        }
        self.v={k:tk.StringVar(value=value) for k,value in defaults.items()}
        self.device=tk.StringVar(value='Системное устройство по умолчанию')
        self.status=tk.StringVar(value='Готово. Вычисление и экспорт WAV работают без звукового устройства.')
        self.result=None;self.state=None;self.recovery=None;self.direct=None;self.current_method=None
        self.devices=[]
        self._computed_settings=None
        self._computed_core=None
        self._setting_timing_preset=False
        for key in ('time_scale','pulse_period','duration','repeat_pulses'):
            self.v[key].trace_add('write',self._timing_was_edited)
        self._build()
        self._refresh_devices()
        self.after(150,self.calculate)
        self.protocol('WM_DELETE_WINDOW',self._close)

    def _style(self):
        sty=ttk.Style(self)
        if 'clam' in sty.theme_names():sty.theme_use('clam')
        self.configure(background='#edf2f7')
        sty.configure('.',font=('Segoe UI',10),background='#edf2f7',foreground='#223449')
        sty.configure('TFrame',background='#edf2f7')
        sty.configure('TLabel',background='#edf2f7')
        sty.configure('Header.TLabel',font=('Segoe UI Semibold',15),foreground='#173a60')
        sty.configure('Sub.TLabel',font=('Segoe UI',9),foreground='#56677b')
        sty.configure('TButton',padding=(9,6))
        sty.configure('TNotebook',background='#edf2f7')
        sty.configure('TNotebook.Tab',padding=(12,8),font=('Segoe UI',10))
        sty.configure('TLabelframe',background='#edf2f7')
        sty.configure('TLabelframe.Label',background='#edf2f7',font=('Segoe UI Semibold',10))
        sty.configure('TEntry',fieldbackground='white')
        sty.configure('TCombobox',fieldbackground='white')

    def _entry(self,parent,label,name):
        row=ttk.Frame(parent)
        row.pack(fill='x',pady=2)
        ttk.Label(row,text=label).pack(side='left')
        ttk.Entry(row,textvariable=self.v[name],width=10,justify='right').pack(side='right',padx=(5,1))

    def _choice(self,parent,label,name,choices):
        ttk.Label(parent,text=label).pack(anchor='w',pady=(3,2))
        cb=ttk.Combobox(parent,textvariable=self.v[name],state='readonly',values=choices)
        cb.pack(fill='x',pady=(0,5))
        return cb

    def _build(self):
        wrapper=ttk.Frame(self,padding=(10,7))
        wrapper.pack(fill='both',expand=True)
        top=ttk.Frame(wrapper);top.pack(fill='x',pady=(0,5))
        ttk.Label(top,text='TAISIA',style='Header.TLabel').pack(side='left')
        ttk.Label(top,text='  аналитические чистые состояния · без функции Вигнера',style='Sub.TLabel').pack(side='left',padx=(4,0))
        # Reserve a bottom strip *before* laying out the expandable notebook.
        # Otherwise Tk's geometry negotiation can clip the status on small screens.
        footer=ttk.Label(wrapper,textvariable=self.status,style='Sub.TLabel',
                         wraplength=max(480,self.winfo_screenwidth()-120))
        footer.pack(side='bottom',fill='x',pady=(5,0))
        main=ttk.Panedwindow(wrapper,orient='horizontal');main.pack(fill='both',expand=True)
        left=ttk.Frame(main,width=270)
        main.add(left,weight=0)
        right=ttk.Frame(main)
        main.add(right,weight=1)
        # Left controls scroll in their own pane. The graphs remain within the window.
        cv=tk.Canvas(left,highlightthickness=0,background='#edf2f7',width=268)
        scroll=ttk.Scrollbar(left,orient='vertical',command=cv.yview)
        cv.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y')
        cv.pack(side='left',fill='both',expand=True)
        content=ttk.Frame(cv,padding=(5,3,7,10))
        win=cv.create_window((0,0),window=content,anchor='nw')
        content.bind('<Configure>',lambda e:cv.configure(scrollregion=cv.bbox('all')))
        cv.bind('<Configure>',lambda e:cv.itemconfigure(win,width=e.width))
        def on_wheel(e):
            cv.yview_scroll(int(-e.delta/120) if e.delta else 0,'units')
        cv.bind('<Enter>',lambda e:cv.bind_all('<MouseWheel>',on_wheel))
        cv.bind('<Leave>',lambda e:cv.unbind_all('<MouseWheel>'))

        st=ttk.LabelFrame(content,text='1. Чистое квантовое состояние',padding=9)
        st.pack(fill='x',pady=(1,8))
        self._choice(st,'Сценический пресет','preset',list(PRESETS))
        ttk.Button(st,text='Применить пресет и пересчитать',command=self.apply_preset).pack(fill='x',pady=(1,7))
        self._choice(st,'Состояние','kind',KINDS)
        self._entry(st,'Фоковское n','n')
        self._entry(st,'Суперпозиция: n₀','n0')
        self._entry(st,'Суперпозиция: n₁','n1')
        self._entry(st,'Коэффициент |z|','ratio')
        self._entry(st,'Фаза z, рад','phase')
        self._entry(st,'Смещение q₀','q0')
        self._entry(st,'Смещение p₀','p0')
        self._entry(st,'Сжатие a','a')
        self._entry(st,'Чирп b','b')
        self._entry(st,'Добавленных фотонов m','m')
        self._entry(st,'Керр χ, рад','kerr')
        self._entry(st,'q-деформация κ','deformation')
        self._entry(st,'Биномиальная p (0–1)','binomial_p')
        ttk.Label(st,text='Используются только параметры выбранного состояния.',style='Sub.TLabel',wraplength=235).pack(fill='x',pady=(6,0))

        rec=ttk.LabelFrame(content,text='2. Томографическое преобразование',padding=9)
        rec.pack(fill='x',pady=(0,8))
        self._choice(rec,'Источник комплексной волны','method',METHODS)
        self._entry(rec,'Размер фоковского базиса d','dimension')
        self._entry(rec,'Сечение θ, градусы','theta')
        ttk.Label(rec,text='Основной: интеграл w → s(t), без ψ и полной матрицы K. При узлах выбирается t₀ ≠ 0.',style='Sub.TLabel',wraplength=230).pack(fill='x')
        ttk.Button(rec,text='↻  Пересчитать и проверить',command=self.calculate).pack(fill='x',pady=(6,0))

        audio=ttk.LabelFrame(content,text='3. Акустический вывод',padding=9)
        audio.pack(fill='x',pady=(0,8))
        ttk.Label(audio,text='Характер звучания').pack(anchor='w',pady=(2,2))
        timing=ttk.Combobox(audio,textvariable=self.v['timing_profile'],
            state='readonly',values=(TIMING_MANUAL,*TIMING_PROFILES,*SINGLE_PULSE_PROFILES))
        timing.pack(fill='x',pady=(0,5))
        timing.bind('<<ComboboxSelected>>',self.apply_timing_profile)
        ttk.Label(audio,textvariable=self.v['timing_profile'],style='Sub.TLabel',
                  wraplength=228).pack(fill='x',pady=(0,4))
        self._entry(audio,'Длительность, с','duration')
        self._entry(audio,'Масштаб T, с','time_scale')
        ttk.Checkbutton(audio,text='Повторять пакеты (сцена)',
                        variable=self.v['repeat_pulses'],onvalue='1',offvalue='0').pack(anchor='w',pady=3)
        self._entry(audio,'Период событий, с','pulse_period')
        self._entry(audio,'Несущая fс, Гц','carrier_hz')
        self._entry(audio,'Громкость (0–0.9)','gain')
        self._choice(audio,'Кодировка','mode',MODES)
        ttk.Label(audio,text=('Слышимое I/Q: L/R — две компоненты комплексного сигнала '
            'после переноса на несущую; обратимо при известной несущей. '
            'Исходное I/Q может казаться тишиной: его частоты слишком низкие. '
            'Повторы — композиционная обработка.'),
            style='Sub.TLabel',wraplength=235).pack(fill='x')

        out=ttk.LabelFrame(content,text='4. Воспроизведение и экспорт',padding=9)
        out.pack(fill='x',pady=(0,3))
        self.devicebox=ttk.Combobox(out,textvariable=self.device,state='readonly',values=[])
        self.devicebox.pack(fill='x',pady=3)
        ttk.Button(out,text='Обновить аудиоустройства',command=self._refresh_devices).pack(fill='x',pady=2)
        ttk.Button(out,text='▶ Воспроизвести',command=self.play).pack(fill='x',pady=2)
        ttk.Button(out,text='■ Остановить',command=self.stop).pack(fill='x',pady=2)
        ttk.Button(out,text='♪ Тест устройства (440 Гц)',command=self.test_tone).pack(fill='x',pady=2)
        ttk.Button(out,text='Сохранить WAV + параметры JSON',command=self.export).pack(fill='x',pady=2)

        notebook=ttk.Notebook(right)
        notebook.pack(fill='both',expand=True,padx=(8,0))
        self.tabs={}
        for key,label in [('tomo','Томограммы'),('kernel','Прямое обращение'),('signal','Сигнал и звук'),('diagnostics','Проверки')]:
            pane=ttk.Frame(notebook,padding=(5,7,5,3));notebook.add(pane,text=label)
            self.tabs[key]=pane
        self.fig_tomo=Figure(figsize=(7.4,5.7),dpi=95,facecolor='#ffffff')
        self.fig_kernel=Figure(figsize=(7.4,5.7),dpi=95,facecolor='#ffffff')
        self.fig_signal=Figure(figsize=(7.4,5.7),dpi=95,facecolor='#ffffff')
        self.can_tomo=FigureCanvasTkAgg(self.fig_tomo,master=self.tabs['tomo'])
        self.can_kernel=FigureCanvasTkAgg(self.fig_kernel,master=self.tabs['kernel'])
        self.can_signal=FigureCanvasTkAgg(self.fig_signal,master=self.tabs['signal'])
        self.can_tomo.get_tk_widget().pack(fill='both',expand=True)
        self.can_kernel.get_tk_widget().pack(fill='both',expand=True)
        self.can_signal.get_tk_widget().pack(fill='both',expand=True)
        self.diag=tk.Text(self.tabs['diagnostics'],wrap='word',font=('Consolas',10),
                          foreground='#233a50',background='white',borderwidth=0,padx=15,pady=15)
        self.diag.pack(fill='both',expand=True)
        self.diag.insert('1.0','Выполните расчёт, чтобы показать диагностику.')
        self.diag.configure(state='disabled')

    def _parse(self):
        def number(key):return float(self.v[key].get().strip().replace(',','.'))
        def integer(key):
            v=number(key)
            if not v.is_integer():raise ValueError(f'{key} должно быть целым числом')
            return int(v)
        state=State(kind=self.v['kind'].get(),n=integer('n'),n0=integer('n0'),
            n1=integer('n1'),ratio=number('ratio'),phase=number('phase'),
            q0=number('q0'),p0=number('p0'),a=number('a'),b=number('b'),
            m=integer('m'),kerr=number('kerr'),deformation=number('deformation'),
            binomial_p=number('binomial_p'))
        state.check()
        theta=number('theta')
        if not 0<=theta<=180:raise ValueError('Угол θ должен лежать от 0° до 180°')
        d=integer('dimension')
        if not 2<=d<=20:raise ValueError('Фоковская размерность: 2–20')
        return state,theta,d

    def _error(self,where,exc):
        self.status.set(f'Ошибка ({where}): {exc}')
        messagebox.showerror('TAISIA: некорректный ввод или ошибка',f'{where}:\n{exc}')

    def _timing_was_edited(self,*_):
        # Any manual T / period change detaches the drop-down from the preset.
        if not self._setting_timing_preset:
            self.v['timing_profile'].set(TIMING_MANUAL)

    def apply_timing_profile(self,event=None):
        name=self.v['timing_profile'].get()
        if name not in TIMING_PROFILES and name not in SINGLE_PULSE_PROFILES:
            return
        self._setting_timing_preset=True
        try:
            if name in SINGLE_PULSE_PROFILES:
                t_value,duration=SINGLE_PULSE_PROFILES[name]
                self.v['time_scale'].set(t_value)
                self.v['duration'].set(duration)
                self.v['repeat_pulses'].set('0')
            else:
                t_value,period=TIMING_PROFILES[name]
                self.v['time_scale'].set(t_value)
                self.v['pulse_period'].set(period)
                self.v['repeat_pulses'].set('1')
        finally:
            self._setting_timing_preset=False
        self.calculate()

    def apply_preset(self):
        values=PRESETS.get(self.v['preset'].get())
        if values is None:return
        for k,v in values.items():self.v[k].set(v)
        self.v['repeat_pulses'].set('1')
        self.v['timing_profile'].set(TIMING_MANUAL)
        self.v['method'].set(METHODS[0])
        self.calculate()

    def calculate(self):
        try:
            current=tuple((k,v.get()) for k,v in self.v.items() if k not in ('preset','timing_profile'))
            if self.result is not None and current==self._computed_settings:return True
            state,theta,d=self._parse()
            method=self.v['method'].get()
            if method not in METHODS:raise ValueError('Неизвестный способ вычисления')
            core=(state,method,d)
            # Acoustic changes (T, duration, carrier, gain...) must never rerun the
            # expensive tomographic inverse if the source quantum state is unchanged.
            core_changed=core!=self._computed_core
            if core_changed:
                self.recovery=None;self.direct=None
            if core_changed and method==METHODS[0]:
                self.status.set('Численное обращение томограмм → s(t)…')
                self.update_idletasks()
                self.direct=recover_direct(state)
                r=self.direct
                if (abs(r.norm-1.)>.02 or r.density_relative_error>.01 or
                    r.kernel_rank_consistency>.05 or r.residual_far_boundary>.08 or
                    r.reference_density_error>.03):
                    raise ValueError('Прямое обращение не прошло контроль точности. Уменьшите параметры смещения, высокие n или увеличьте точность численной сетки.')
            if core_changed and method==METHODS[1]:
                self.recovery=recover(state,dimension=d)
                if self.recovery.tomographic_relative_error > .01:
                    raise ValueError('Томографическая реконструкция неточна (E > 1%). Увеличьте d или уменьшите |q₀|, |p₀|, сжатие/чирп.')
            self.state=state;self.current_method=method
            self.result=self._render_audio()
            self._draw(state,theta)
            self._diagnose(state,theta)
            self._computed_settings=current
            self._computed_core=core
            warn=(' Энергия в окне < 99%: увеличьте длительность или измените T.'
                  if self.result.captured_energy<.99 else '')
            self.status.set(f'Готово: {state.kind}. ∫окно |s|²dt = {self.result.captured_energy:.6f}; '
                            f'пик аудио = {self.result.audio_peak:.3f}.{warn}')
            return True
        except (ValueError,TypeError,FloatingPointError,OverflowError,MemoryError,np.linalg.LinAlgError) as exc:
            self._error('Расчёт',exc)
            return False
        except Exception as exc:
            self.status.set(f'Непредвиденная ошибка: {exc}')
            self._error('Расчёт',exc)
            traceback.print_exc()
            return False

    def _render_audio(self):
        def num(k):return float(self.v[k].get().replace(',','.'))
        coeff=None if self.recovery is None else self.recovery.coeff
        return render(self.state,duration=num('duration'),time_scale=num('time_scale'),
            carrier_hz=num('carrier_hz'),gain=num('gain'),mode=self.v['mode'].get(),coeff=coeff,
            sampled_q=(self.direct.q if self.direct is not None else None),
            sampled_psi=(self.direct.signal_q if self.direct is not None else None),
            repeat_pulses=(self.v['repeat_pulses'].get()=='1'),
            pulse_period=num('pulse_period'))

    def _draw(self,state,theta):
        x=np.linspace(-8.5,8.5,440)
        th=np.linspace(0,np.pi,91)
        W=np.array([tomogram(state,x,float(t)) for t in th])
        recovered=None
        if self.recovery is not None:
            recovered=wave_from_coeff(self.recovery.coeff,x)
        if self.direct is not None:
            recovered=CubicSpline(self.direct.q,self.direct.signal_q)(x)

        # Tomogram pane: 2D family and selected cross-section
        fig=self.fig_tomo;fig.clear()
        axes=fig.subplots(2,1,gridspec_kw={'height_ratios':[1.28,.8]})
        ax,ax2=axes
        ax.imshow(W,origin='lower',aspect='auto',extent=(x[0],x[-1],0,180),cmap='viridis')
        ax.axhline(theta,color='white',lw=.8,alpha=.75,ls='--')
        ax.set(xlabel='Квадратура X',ylabel='θ, градусы',title='Оптические томограммы w(X | θ)')
        target=tomogram(state,x,np.deg2rad(theta))
        ax2.plot(x,target,lw=1.7,label='Аналитическая w(X | θ)')
        if recovered is not None:
            if self.recovery is not None:
                phi=basis(x,len(self.recovery.coeff))
                tr=(self.recovery.coeff*np.exp(-1j*np.arange(len(self.recovery.coeff))*np.deg2rad(theta))) @ phi
                ax2.plot(x,np.abs(tr)**2,lw=1.2,ls='--',label='Фоковская реконструкция',alpha=.9)
            else:
                predicted=optical_from_wave(self.direct.q,self.direct.signal_q,x,np.deg2rad(theta))
                if predicted is not None:
                    ax2.plot(x,predicted,lw=1.2,ls='--',label='Прямая ψ → w (Френель)',alpha=.9)
        ax2.set(xlabel='X',ylabel='Вероятностная плотность')
        ax2.grid(alpha=.18)
        ax2.legend(fontsize=8,loc='best')
        fig.tight_layout(pad=1.5)
        self.can_tomo.draw_idle()

        # The displayed quantity is the reconstructed complex SIGNAL s(Tq)
        # (in the dimensionless scale T=1 for plotting), not a separate ψ stage.
        fig=self.fig_kernel;fig.clear()
        if self.direct is not None:
            r=self.direct
            ax1,ax2=fig.subplots(2,1)
            ax1.plot(r.q,r.signal_q.real,label='Re s(Tq) · √T',lw=1.5)
            ax1.plot(r.q,r.signal_q.imag,label='Im s(Tq) · √T',lw=1.2)
            ax1.axvline(r.reference_q,ls='--',lw=.85,color='gray',label='q_ref')
            ax1.set(title='Прямое восстановление s из w по интегралу Манько',xlabel='q=t/T',ylabel='Комплексная амплитуда')
            ax1.grid(alpha=.2);ax1.legend(fontsize=8)
            ax2.plot(r.q,abs(r.signal_q)**2,label='|s(Tq)|² · T',lw=1.4)
            ax2.plot(r.q,tomogram(state,r.q,0),ls='--',label='w(q|1,0)',lw=1.3)
            ax2.set(title='Проверка плотности восстановленного сигнала',xlabel='q=t/T',ylabel='Плотность')
            ax2.grid(alpha=.2);ax2.legend(fontsize=8)
        else:
            ax1=fig.subplots(1,1)
            ax1.text(.5,.5,'Доступно в режиме «Прямое: w → s»',
                     ha='center',va='center',transform=ax1.transAxes,fontsize=12)
            ax1.axis('off')
        fig.tight_layout(pad=1.5)
        self.can_kernel.draw_idle()

        # Signal pane: recovered complex signal, then acoustic output.
        fig=self.fig_signal;fig.clear()
        ax,ax2=fig.subplots(2,1,gridspec_kw={'height_ratios':[1,1]})
        qgrid=np.linspace(-8.5,8.5,550)
        if self.direct is not None:
            wave_q=CubicSpline(self.direct.q,self.direct.signal_q)(qgrid)
        elif self.recovery is not None:
            wave_q=wave_from_coeff(self.recovery.coeff,qgrid)
        else:
            wave_q=wave(state,qgrid)
        wave_q=phase_fix(wave_q)
        ax.plot(qgrid,np.real(wave_q),label='Re s(Tq) · √T',lw=1.3)
        ax.plot(qgrid,np.imag(wave_q),label='Im s(Tq) · √T',lw=1.1)
        ax.plot(qgrid,np.abs(wave_q)**2,label='|s(Tq)|² · T',lw=1.4)
        ax.set(xlabel='q = t/T',title='Комплексный сигнал из томограмм (масштаб T=1)')
        ax.legend(loc='best',fontsize=8,ncol=3);ax.grid(alpha=.2)
        r=self.result
        down=max(1,len(r.seconds)//6000)
        smp=r.samples[::down]
        if smp.ndim==1:ax2.plot(r.seconds[::down],smp,lw=.6,label='Carrier (mono)')
        else:
            encoding='s·exp(i2πfct)' if r.mode == MODE_IQ_AUDIBLE else 's'
            ax2.plot(r.seconds[::down],smp[:,0],lw=.6,label=f'I = Re({encoding})')
            ax2.plot(r.seconds[::down],smp[:,1],lw=.6,label=f'Q = Im({encoding})',alpha=.7)
        ax2.set(xlabel='Время, с',ylabel='Амплитуда PCM',title='Готовая аудиоволна (после акустического отображения)')
        ax2.grid(alpha=.2);ax2.legend(fontsize=8,loc='upper right')
        fig.tight_layout(pad=1.5)
        self.can_signal.draw_idle()

    def _diagnose(self,state,theta):
        x=np.linspace(-12,12,4801)
        integrals=[float(trapezoid(tomogram(state,x,t),x=x)) for t in (0,np.pi/4,np.pi/2)]
        w=tomogram(state,x,np.deg2rad(theta))
        squared=float(trapezoid(np.abs(wave(state,x))**2,x=x))
        details=[
            'TAISIA 0.5 — контроль математической корректности','',
            '[Соглашения]',
            'ħ=1, [q̂,p̂]=i, α=(q₀+i p₀)/√2',
            'w(X|θ)=Tr[ρ δ(X−q̂cosθ−p̂sinθ)]',
            'M(X|α,β)=w(X|Tα,β/T). Аудио-время отсчитывается от центра события.',
            '', '[Нормировки]',
            f'∫ |ψ(q)|² dq          = {squared:.9f}  (q∈[-12,12])',
            f'∫ w(X | 0°) dX        = {integrals[0]:.9f}',
            f'∫ w(X | 45°) dX       = {integrals[1]:.9f}',
            f'∫ w(X | 90°) dX       = {integrals[2]:.9f}',
            f'min_X w(X|{theta:g}°)  = {np.min(w):.3g}',
            f'∫_аудиоокно |s(t)|²dt = {self.result.captured_energy:.9f}',
            '', '[Восстановление из томограмм]',
        ]
        if self.direct is not None:
            r=self.direct
            details += [
                'ИСТОЧНИК АУДИО: НЕПОСРЕДСТВЕННАЯ ФОРМУЛА w → s(t)!',
                's(t)=[2πT√M(t_ref|1,0)]⁻¹∫dμdX w(X|μ,(t-t_ref)/T)exp[iX−iμ(t+t_ref)/(2T)].',
                'Точка t_ref автоматически выбирается по максимуму w(q|1,0); при t_ref=0 формула совпадает с указанной пользователем.',
                'В режиме Gaussian внутренняя X-Фурье интегрируется точно из трёх значений плотности w.',
                'Остальные состояния: численная гаусс-эрмитова квадратура по X.',
                f'Опорный q_ref={r.reference_q:.6f}; ошибка диагонали={r.reference_density_error:.3e}',
                f'∫ |s_q|²dq = {r.norm:.9f}',
                f'Относительная ошибка плотности |s_q|² = {r.density_relative_error:.3e}',
                f'Согласованность второго столбца K (rank 1) = {r.kernel_rank_consistency:.3e}',
                f'Фиделити с независимым аналитическим эталоном = {r.fidelity_reference_only:.9f}',
                f'Остаток на краях q-окна = {r.residual_far_boundary:.3e}',
                f'μ_макс = ±{r.mu_max:.2f}; шаг μ={r.mu_step:.5f}; узлы интеграла X={r.quadrature_nodes}',
                'Сигнал s(t) вычислен непосредственно из w; волновая функция не реконструируется,',
                'интерполяция применяется только при аудиодискретизации.',
            ]
        elif self.recovery is None:
            details += ['Источник: аналитическая ψ(q) (эталон, без численной инверсии).']
        else:
            r=self.recovery
            details += [
                'Источник: ЧИСЛЕННАЯ инверсия w(X|θ) в базисе Эрмита.',
                f'Размерность d = {r.dimension}, N_X = {r.points}, N_θ = {r.angles}.',
                f'Старшее собственное значение восстановленной ρ = {r.largest_eigenvalue:.9f}',
                f'Сумма отрицательных собственных значений по модулю = {r.negative_eigenmass:.3e}',
                f'Относительная квадратичная ошибка томограмм = {r.tomographic_relative_error:.3e}',
                'Конечное d может давать неточную, не положительную ρ.',
                'Звуковая волна берётся из ведущего собственного вектора ρ.',
            ]
        details += [
            '', '[Аудио]',
            f'Кодировка: {self.result.mode}',
            f'Частота: {self.result.samplerate} Гц (PCM16 при экспорте)',
            f'Несущая: {self.result.carrier_hz:g} Гц',
            f'Масштаб: T={self.result.time_scale:g} с',
            f'Сценические повторы: {self.result.repeat_pulses}, период {self.result.pulse_period:g} с',
            'Повторы не являются унитарным образом одиночного квантового состояния.',
            f'Пик: {self.result.audio_peak:.4f}; RMS: {self.result.audio_rms:.4f}',
            'Нормировка по пику и краевой fade 5 мс НЕ являются унитарными.',
            'Томограмма комплексной огибающей ≠ томограмма вещественного аудио.',
            '', '[Границы версии]',
            'Только одна непрерывная степень свободы и чистые состояния.',
            'Нет экспериментальной гомодинной выборки, смесей, OSC и MIDI.',
            'Аналитическое представление проверяется в конечном q/X-окне.',
            'Эта сборка не включает приложение QHOSYN.',
        ]
        self.diag.configure(state='normal')
        self.diag.delete('1.0','end')
        self.diag.insert('1.0','\n'.join(details))
        self.diag.configure(state='disabled')

    def _refresh_devices(self):
        previous=self.device.get()
        self.devices=[('Системное устройство по умолчанию',None)]
        if sd is not None:
            try:
                for i,device in enumerate(sd.query_devices()):
                    if device['max_output_channels']>0:
                        self.devices.append((f'{i}: {device["name"]}',i))
            except Exception as exc:self.status.set(f'Аудиоустройства: {exc}')
        self.devicebox.configure(values=[a for a,b in self.devices])
        self.device.set(previous if previous in [a for a,b in self.devices] else self.devices[0][0])

    def _device_index(self):
        return next((v for label,v in self.devices if label==self.device.get()),None)

    def play(self):
        if not self.calculate():return
        if sd is None:
            self._error('Воспроизведение',RuntimeError('Пакет sounddevice не установлен или PortAudio недоступен. WAV-экспорт работает отдельно.'))
            return
        try:
            if self.result.samples.ndim==2:
                out_device=sd.query_devices(self._device_index(),kind='output')
                if out_device['max_output_channels'] < 2:
                    raise ValueError('Выбранное устройство поддерживает только моно. '
                                     'Для Stereo I/Q выберите стереоустройство или Carrier (mono).')
            sd.stop()
            sd.play(self.result.samples,samplerate=self.result.samplerate,
                    device=self._device_index(),blocking=False)
            self.status.set(('Слышимый Stereo I/Q: обе компоненты с несущей. '
                'Исходное I/Q может казаться тишиной.' if self.result.mode == MODE_IQ_AUDIBLE else
                'Воспроизведение запущено. Для проверки оборудования есть тест 440 Гц.'))
        except Exception as exc:self._error('Аудиоустройство',exc)

    def stop(self):
        if sd is not None:
            try:sd.stop()
            except Exception as exc:self.status.set(f'Остановка устройства: {exc}')
        self.status.set('Воспроизведение остановлено.')

    def test_tone(self):
        if sd is None:
            self._error('Аудиотест',RuntimeError('Установите sounddevice для воспроизведения.'))
            return
        try:
            fs=48000;t=np.arange(fs)/fs
            # Independent of the quantum module; suitable for audio device diagnosis.
            signal=.16*np.sin(2*np.pi*440*t)*np.minimum(1,np.minimum(t/.02,(1-t)/.02))
            sd.stop();sd.play(signal.astype(np.float32),samplerate=fs,device=self._device_index(),blocking=False)
            self.status.set('Тест: 1 секунда, синусоида 440 Гц, независимо от TAISIA.')
        except Exception as exc:self._error('Аудиотест',exc)

    def export(self):
        if not self.calculate():return
        path=filedialog.asksaveasfilename(title='Сохранить TAISIA WAV',defaultextension='.wav',
            filetypes=[('Wave audio','*.wav')],initialfile='TAISIA_pure_state.wav')
        if not path:return
        try:
            wav_path,json_path=save_wav(path,self.result,self.state,
                source=('Direct Manko integral w -> s(t)' if self.direct is not None
                        else 'Fock inversion from analytic tomograms' if self.recovery is not None
                        else 'analytic wavefunction'),
                diagnostics=({'relative_tomogram_squared_error':self.recovery.tomographic_relative_error,
                              'fock_dimension':self.recovery.dimension,
                              'max_density_eigenvalue':self.recovery.largest_eigenvalue} if self.recovery
                             else {'relative_density_squared_error':self.direct.density_relative_error,
                                   'kernel_rank_consistency':self.direct.kernel_rank_consistency,
                                   'fidelity_to_analytic_reference':self.direct.fidelity_reference_only,
                                   'direct_mu_max':self.direct.mu_max,
                                   'reference_q':self.direct.reference_q} if self.direct else None))
            self.status.set(f'Сохранены {wav_path.name} и {json_path.name}.')
            messagebox.showinfo('TAISIA','WAV и JSON записаны:\n'+str(wav_path)+'\n'+str(json_path))
        except Exception as exc:self._error('Запись WAV',exc)

    def _close(self):
        if sd is not None:
            try:sd.stop()
            except Exception:pass
        self.destroy()


if __name__=='__main__':
    App().mainloop()
