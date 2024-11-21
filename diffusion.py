from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from ase.io import read
from ase.units import J, mol

class VolumeCalculator:
    """POSCAR文件体积计算器"""
    
    @staticmethod
    def read_vasp_volume(self, vasp_file: Path) -> float:
        """
        读取单个VASP文件并计算体积
        
        Args:
            vasp_file: .vasp文件路径
            
        Returns:
            体积(cm^3)
        """
        try:
            with open(vasp_file, 'r') as f:
                lines = f.readlines()
            
            scaling_factor = float(lines[1].strip())
            
            # 读取晶格向量
            lattice_vectors = []
            for i in range(2, 5):
                vector = list(map(float, lines[i].split()))
                lattice_vectors.append(vector)
            
            lattice_vectors = np.array(lattice_vectors) * scaling_factor
            volume = np.abs(np.linalg.det(lattice_vectors))
            
            # 转换单位: Å^3 -> cm^3
            volume_cm3 = volume * 1e-24
            return volume_cm3
            
        except Exception as e:
            self.logger.error(f"处理文件 {vasp_file.name} 时出错: {str(e)}")
            return None

class DiffusionAnalyzer:
    """扩散系数和质子电导率分析器"""
    
    def __init__(self):
        # 物理常数
        self.e = 1.60217662e-19  # 基本电荷(C)
        self.k_B = 1.38064852e-23  # 玻尔兹曼常数(J/K)
        
    def calculate_conductivity(
        self,
        n_protons: int,
        diffusion_coef: float,
        volume: float,
        temperature: float
    ) -> float:
        """
        计算质子电导率
        
        Args:
            n_protons: 质子数量
            diffusion_coef: 扩散系数(cm^2/s)
            volume: 体积(cm^3)
            temperature: 温度(K)
            
        Returns:
            电导率(S/cm)
        """
        sigma = (n_protons * self.e**2 * diffusion_coef) / (volume * self.k_B * temperature)
        return sigma
        
    def analyze_trajectory(
        self,
        traj_file: str,
        temperature: float,
        atom_type: str = 'H',
        shift_t: int = 500,
        window_size: int = 1000,
        step: int = 1
    ) -> Dict:
        """
        分析轨迹文件计算MSD和扩散系数
        
        Args:
            traj_file: 轨迹文件路径
            temperature: 温度
            atom_type: 原子类型
            shift_t: 时间位移
            window_size: 时间窗口大小
            step: 计算步长
            
        Returns:
            分析结果字典
        """
        # 读取轨迹
        traj_list = read(traj_file, index=":")
        atom_index = [i for i, x in enumerate(traj_list[0].get_chemical_symbols()) 
                     if x == atom_type]
                     
        # 计算体积
        volume = [atoms.get_volume() for atoms in traj_list]
        volume_cm3 = np.mean(volume) * 1e-24
        
        # 提取位置数据
        positions_all = np.array([traj_list[i].get_positions() 
                                for i in range(len(traj_list))])
        positions = positions_all[:, atom_index]
        
        # 计算MSD和扩散系数
        msd_list = []
        D_list = []
        
        for i in range(0, int(len(positions)/shift_t)):
            msd_t = np.mean(np.sum(
                (positions[i*shift_t:i*shift_t + window_size] - 
                 positions[i*shift_t])**2, axis=2), axis=1)
                 
            if len(msd_t) != window_size:
                continue
                
            slope, intercept = np.polyfit(
                range(0, window_size, step),
                msd_t[::step],
                1
            )
            
            D = slope / 6
            D_list.append(D)
            msd_list.append(msd_t[::step])
            
        # 计算平均值
        D_avg = np.mean(D_list)
        msd_avg = np.mean(msd_list, axis=0)
        
        # 单位转换: Å^2/ps -> cm^2/s
        D_cm2_s = D_avg * 1e-16 / 1e-12
        
        # 计算电导率
        sigma = self.calculate_conductivity(
            n_protons=len(atom_index),
            diffusion_coef=D_cm2_s,
            volume=volume_cm3,
            temperature=temperature
        )
        
        return {
            'T(K)': temperature,
            '1000/T': 1000/temperature,
            'D(cm^2/s)': D_cm2_s,
            'log10_D': np.log10(D_cm2_s),
            'sigma(S/cm)': sigma,
            'log10_sigma': np.log10(sigma),
            'msd': msd_avg,
            'volume': volume_cm3
        }

class PlotManager:
    """绘图管理器"""
    
    def __init__(self, font_size: int = 26):
        self.font_size = font_size
        self._setup_style()
        
    def _setup_style(self):
        """设置绘图风格"""
        plt.rcParams.update({
            'font.size': self.font_size,
            'font.family': 'DejaVu Sans',
            'legend.fontsize': self.font_size,
            'xtick.labelsize': self.font_size,
            'ytick.labelsize': self.font_size,
            'axes.labelsize': self.font_size,
            'axes.titlesize': self.font_size,
            'figure.figsize': [14, 14]
        })
        
    def plot_arrhenius(
        self,
        data: pd.DataFrame,
        save_dir: str,
        column_name: str = 'log10_D',
        y_label: str = 'log[D(cm$^2$ sec$^{-1}$)]',
        show_ea: bool = True
    ):
        """
        绘制Arrhenius图
        
        Args:
            data: 数据DataFrame
            save_dir: 保存目录
            column_name: 要绘制的列名
            y_label: y轴标签
            show_ea: 是否显示激活能
        """
        fig, ax1 = plt.subplots()
        
        # 主坐标轴: 1000/T
        slope, intercept, r_value, _, _ = stats.linregress(
            data["1000/T"],
            data[column_name]
        )
        
        ax1.set_xlabel('1000/T [K$^{-1}$]')
        ax1.set_ylabel(y_label)
        ax1.scatter(data["1000/T"], data[column_name], color='k', linewidth=4)
        ax1.plot(
            data["1000/T"],
            data["1000/T"]*slope + intercept,
            '--k',
            linewidth=4
        )
        
        # 次坐标轴: 温度
        ax2 = ax1.twiny()
        ax2.set_xlabel('Temperature (K)')
        ax2.plot(data['T(K)'], data[column_name], 'k')
        ax2.invert_xaxis()
        ax2.lines[0].set_visible(False)
        
        if show_ea:
            R = 8.31446261815324  # 气体常数
            E_act = -slope * 1000 * np.log(10) * R * (J / mol)
            ax1.text(
                data["1000/T"].iloc[1],
                data[column_name].iloc[-2],
                f'Ea: {E_act:.2f} eV',
                fontsize=self.font_size,
                color='red'
            )
            
        plt.tight_layout()
        plt.savefig(
            Path(save_dir) / f'arrhenius_{column_name}.png',
            bbox_inches='tight',
            pad_inches=0.3,
            dpi=300
        )
        plt.close()
        
    def plot_diffusion_coefficient(
        self,
        data: pd.DataFrame,
        save_dir: str,
        show_ea: bool = True
    ):
        """
        绘制扩散系数图
        
        Args:
            data: 数据DataFrame
            save_dir: 保存目录
            show_ea: 是否显示激活能
        """
        fig, ax1 = plt.subplots()
        
        ax1.set_xlabel('1000/T [K$^{-1}$]')
        ax1.set_ylabel('D(cm$^2$ sec$^{-1}$)')
        ax1.plot(data['1000/T'], data['D(cm^2/s)'], 'k', linewidth=4)
        
        ax2 = ax1.twiny()
        ax2.set_xlabel('Temperature (K)')
        ax2.plot(data['T(K)'], data['D(cm^2/s)'], 'k')
        ax2.invert_xaxis()
        ax2.lines[0].set_visible(False)
        
        if show_ea:
            slope, intercept, _, _, _ = stats.linregress(
                data["1000/T"],
                data["log10_D"]
            )
            R = 8.31446261815324
            E_act = -slope * 1000 * np.log(10) * R * (J / mol)
            ax1.text(
                data["1000/T"].iloc[1],
                data['D(cm^2/s)'].iloc[-2],
                f'Ea: {E_act:.2f} eV',
                fontsize=self.font_size,
                color='red'
            )
            
        plt.tight_layout()
        plt.savefig(
            Path(save_dir) / 'diffusion_coefficient.png',
            bbox_inches='tight',
            pad_inches=0.3,
            dpi=300
        )
        plt.close()

def main():
    """使用示例"""
    # 初始化分析器
    volume_calc = VolumeCalculator()
    diffusion_analyzer = DiffusionAnalyzer()
    plot_manager = PlotManager()
    
    # 分析目录和温度设置
    working_dir = Path("md_analysis")
    working_dir.mkdir(exist_ok=True)
    
    temperatures = [300, 400, 500, 600]
    results = []
    
    # 分析每个温度的轨迹
    for temp in temperatures:
        traj_file = f"MD_{temp}.traj"
        if not Path(traj_file).exists():
            print(f"找不到轨迹文件: {traj_file}")
            continue
            
        result = diffusion_analyzer.analyze_trajectory(
            traj_file=traj_file,
            temperature=temp
        )
        results.append(result)
        
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(working_dir / 'diffusion_results.csv', index=False)
    
    # 绘制图形
    plot_manager.plot_arrhenius(df, str(working_dir))
    plot_manager.plot_diffusion_coefficient(df, str(working_dir))

if __name__ == "__main__":
    main()