import argparse
import json
import sys
import os
import pickle
import gzip
import multiprocessing as mp
from pathlib import Path
import lmdb
import numpy as np
import tqdm
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

# 禁用 RDKit 警告
RDLogger.DisableLog('rdApp.*')

# 禁用 Python 输出缓冲
sys.stdout = sys.stdout
sys.stderr = sys.stderr


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DrugCLIP 虚拟筛选预测脚本')
    parser.add_argument('-i', '--input', type=str, required=True, help='输入 JSON 文件路径')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出目录路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点目录')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--dataset-name', type=str, default="dude", help='数据集名称（用于 LMDB 路径）')
    
    # test.sh 相关参数
    parser.add_argument('--results-path', type=str, default='./test', help='test.sh 的结果输出路径')
    # 移除 test-task 参数,现在使用通用推理
    # parser.add_argument('--test-task', type=str, default='DUDE', choices=['PCBA', 'DUDE'], help='test.sh 的测试任务')
    parser.add_argument('--gpu-id', type=str, default='1', help='test.sh 使用的 GPU ID')
    
    return parser.parse_args()


def load_input_data(input_path):
    """加载输入数据"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def predict(data_item, checkpoint_dir, batch_size):
    """
    执行预测
    
    Args:
        data_item: 单个输入数据项
        checkpoint_dir: 模型检查点目录
        batch_size: 批次大小
        
    Returns:
        预测结果字典
    """
    # TODO: 在此处添加实际的模型预测逻辑
    # 当前为示例实现,返回随机分数
    import random
    score = random.uniform(0.0, 1.0)
    
    return {
        'name': data_item['name'],
        'score': score
    }


def save_output(output_dir, name, result):
    """保存预测结果"""
    output_path = Path(output_dir) / f"{name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)


# ========== 数据转换函数（从 write_dude_multi.py 改编）==========

def gen_conformation(mol, num_conf=20, num_worker=8):
    """生成分子构象"""
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, numThreads=num_worker, 
                                   pruneRmsThresh=1, maxAttempts=10000, useRandomCoords=False)
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker)
        except:
            pass
        mol = Chem.RemoveHs(mol)
    except:
        print(f"无法生成构象: {Chem.MolToSmiles(mol)}", flush=True)
        return None
    if mol.GetNumConformers() == 0:
        print(f"无法生成构象: {Chem.MolToSmiles(mol)}", flush=True)
        return None
    return mol


def convert_mol_to_data(mol, num_conf=1, num_worker=5):
    """将分子转换为数据格式"""
    mol = gen_conformation(mol, num_conf, num_worker)
    if mol is None:
        return None
    coords = [np.array(mol.GetConformer(i).GetPositions()) for i in range(mol.GetNumConformers())]
    atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
    return {'coords': coords, 'atom_types': atom_types, 'smi': Chem.MolToSmiles(mol), 'mol': mol}


def convert_mol_to_data_keep_conformation(mol):
    """
    将分子转换为数据格式，保留原始构象（不重新生成）
    用于处理 decoys，因为它们已经有 3D 构象
    
    Args:
        mol: RDKit 分子对象（已包含 3D 构象）
        
    Returns:
        包含坐标、原子类型等信息的字典，如果失败返回 None
    """
    if mol is None:
        return None
    
    try:
        # 检查是否有构象
        if mol.GetNumConformers() == 0:
            print(f"警告: 分子没有构象，跳过: {Chem.MolToSmiles(mol)}", flush=True)
            return None
        
        # 直接使用已有的构象，不重新生成
        coords = [np.array(mol.GetConformer(i).GetPositions()) for i in range(mol.GetNumConformers())]
        atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
        
        return {
            'coords': coords, 
            'atom_types': atom_types, 
            'smi': Chem.MolToSmiles(mol), 
            'mol': mol
        }
    except Exception as e:
        print(f"转换分子失败: {e}", flush=True)
        return None


def read_pdb(path):
    """读取 PDB 文件"""
    print(f"  读取 PDB: {path}", flush=True)
    pdb_df = PandasPdb().read_pdb(path)
    coord = pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']]
    atom_type = pdb_df.df['ATOM']['atom_name']
    residue_name = pdb_df.df['ATOM']['chain_id'] + pdb_df.df['ATOM']['residue_number'].astype(str)
    residue_type = pdb_df.df['ATOM']['residue_name']
    protein = {
        'coord': np.array(coord), 
        'atom_type': list(atom_type),
        'residue_name': list(residue_name),
        'residue_type': list(residue_type)
    }
    return protein


def read_mol2_ligand(path):
    """读取 MOL2 配体文件"""
    print(f"  读取 MOL2: {path}", flush=True)
    mol2_df = PandasMol2().read_mol2(path)
    coord = mol2_df.df[['x', 'y', 'z']]
    atom_type = mol2_df.df['atom_name']
    ligand = {'coord': np.array(coord), 'atom_type': list(atom_type), 'mol': Chem.MolFromMol2File(path)}
    return ligand


def read_sdf(path):
    """读取 SDF 文件"""
    print(f"  读取 SDF: {path}", flush=True)
    suppl = Chem.SDMolSupplier(path, removeHs=True, sanitize=True)
    mols = [mol for mol in suppl if mol is not None]
    print(f"    从 SDF 读取了 {len(mols)} 个分子", flush=True)
    return mols


def add_charges(m):
    """添加电荷（修复化学问题）"""
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    if not ps:
        Chem.SanitizeMol(m)
        return m
    for p in ps:
        if p.GetType()=='AtomValenceException':
            at = m.GetAtomWithIdx(p.GetAtomIdx())
            if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(1)
            if at.GetAtomicNum()==6 and at.GetExplicitValence()==5:
                for b in at.GetBonds():
                    if b.GetBondType()==Chem.rdchem.BondType.DOUBLE:
                        b.SetBondType(Chem.rdchem.BondType.SINGLE)
                        break
            if at.GetAtomicNum()==8 and at.GetFormalCharge()==0 and at.GetExplicitValence()==3:
                at.SetFormalCharge(1)
            if at.GetAtomicNum()==5 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(-1)
    try:
        Chem.SanitizeMol(m)
    except:
        return None
    return m


def read_sdf_gz(path):
    """读取 gzip 压缩的 SDF 文件"""
    print(f"  读取 SDF.GZ: {path}", flush=True)
    inf = gzip.open(path)
    with Chem.ForwardSDMolSupplier(inf, removeHs=False, sanitize=False) as gzsuppl:
        ms = [add_charges(x) for x in gzsuppl if x is not None]
    ms = [rdMolStandardize.Uncharger().uncharge(Chem.RemoveHs(m)) for m in ms if m is not None]
    print(f"    从 SDF.GZ 读取了 {len(ms)} 个分子", flush=True)
    return ms


def get_pocket_residues(protein, ligand, radius=6):
    """获取口袋残基"""
    protein_coord = protein['coord']
    ligand_coord = ligand['coord']
    protein_residue_name = protein['residue_name']
    pocket_residue = set()
    for i in range(len(protein_coord)):
        for j in range(len(ligand_coord)):
            if np.linalg.norm(protein_coord[i] - ligand_coord[j]) < radius:
                pocket_residue.add(protein_residue_name[i])
    return pocket_residue


def write_lmdb(data, lmdb_path):
    """写入 LMDB 数据库"""
    print(f"  写入 LMDB: {lmdb_path} ({len(data)} 条记录)", flush=True)
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, 
                    readahead=False, meminit=False, map_size=1099511627776)
    num = 0
    with env.begin(write=True) as txn:
        for d in data:
            txn.put(str(num).encode('ascii'), pickle.dumps(d))
            num += 1
    env.close()
    print(f"  完成写入 {num} 条记录", flush=True)



def deal_input_data(inputs, dataset_name):
    """处理输入数据并转换为 LMDB 格式"""
    # 加载输入数据
    input_data = load_input_data(inputs)
    print(f"加载了 {len(input_data)} 条输入数据", flush=True)
    
    # 对每个 target 进行处理
    for item in input_data:
        target_name = item['name']
        print(f"\n{'='*80}", flush=True)
        print(f"正在处理目标: {target_name}", flush=True)
        print(f"{'='*80}", flush=True)
        print(item)
        
        # 修改字段名以匹配实际的 dict 结构
        receptor_pdb = item.get('receptor_path')
        pocket_pdb = item.get('pocket_path')
        crystal_ligand = item.get('ligand_path')
        actives_sdf = item.get('actives_path')
        decoys_sdf = item.get('decoys_path')
        
        # 输出目录 - 与输入文件在同一目录
        # 输入文件路径格式: /data/data/{target_name}/xxx.pdb
        # LMDB 也保存到: /data/data/{target_name}/
        target_output_dir = f"/data/data/{target_name}"
        os.makedirs(target_output_dir, exist_ok=True)
        print(f"输出目录: {target_output_dir}", flush=True)
        print(f"  (LMDB 文件将保存在此目录)", flush=True)
        
        # 1. 读取蛋白和配体
        if not receptor_pdb or not crystal_ligand:
            print(f"警告: 缺少必要文件，跳过 {target_name}", flush=True)
            continue
            
        protein = read_pdb(receptor_pdb)
        ligand = read_mol2_ligand(crystal_ligand)
        
        # 2. 生成口袋数据
        print(f"\n生成口袋数据...", flush=True)
        pocket_residues = get_pocket_residues(protein, ligand, radius=6)
        pocket_atom_idx = [i for i, r in enumerate(protein['residue_name']) if r in pocket_residues]
        pocket_atom_type = [protein['atom_type'][i] for i in pocket_atom_idx]
        pocket_coord = [protein['coord'][i] for i in pocket_atom_idx]
        
        pocket_data = {
            'pocket': target_name,
            'pocket_index': 0,
            'pocket_atoms': pocket_atom_type,
            'pocket_coordinates': pocket_coord
        }
        
        pocket_lmdb_path = os.path.join(target_output_dir, 'pocket.lmdb')
        if os.path.exists(pocket_lmdb_path):
            import shutil
            if os.path.isdir(pocket_lmdb_path):
                shutil.rmtree(pocket_lmdb_path)
            else:
                os.remove(pocket_lmdb_path)
            print(f"  - 已删除现有的 LMDB 数据库: {pocket_lmdb_path}", flush=True)
        write_lmdb([pocket_data], pocket_lmdb_path)
        
        # 3. 处理分子数据
        print(f"\n处理分子数据...", flush=True)
        mol_data = []
        
        # 读取活性分子
        if actives_sdf and os.path.exists(actives_sdf):
            print(f"处理活性分子...", flush=True)
            active_mols = read_sdf(actives_sdf)
            print(f"  开始转换 {len(active_mols)} 个活性分子...", flush=True)
            
            pool = mp.Pool(min(32, mp.cpu_count()))
            converted_actives = list(tqdm.tqdm(
                pool.imap_unordered(convert_mol_to_data, active_mols),
                total=len(active_mols),
                desc="  转换活性分子"
            ))
            pool.close()
            pool.join()
            
            converted_actives = [m for m in converted_actives if m is not None]
            print(f"  成功转换 {len(converted_actives)} 个活性分子", flush=True)
            
            for m in converted_actives:
                mol_data.append({
                    'atoms': m['atom_types'],
                    'coordinates': m['coords'],
                    'smi': m['smi'],
                    'mol': m['mol'],
                    'label': 1
                })
        
        # 读取诱饵分子（使用原始构象，不重新生成）
        if decoys_sdf and os.path.exists(decoys_sdf):
            print(f"处理诱饵分子（保留原始构象）...", flush=True)
            if decoys_sdf.endswith('gz'):
                decoy_mols = read_sdf_gz(decoys_sdf)
            else:
                decoy_mols = read_sdf(decoys_sdf)
            
            print(f"  开始转换 {len(decoy_mols)} 个诱饵分子...", flush=True)
            print(f"  注意: 使用 SDF 中的原始 3D 构象，不重新生成", flush=True)
            
            # 使用新函数 convert_mol_to_data_keep_conformation 保留原始构象
            pool = mp.Pool(min(32, mp.cpu_count()))
            converted_decoys = list(tqdm.tqdm(
                pool.imap_unordered(convert_mol_to_data_keep_conformation, decoy_mols),
                total=len(decoy_mols),
                desc="  转换诱饵分子"
            ))
            pool.close()
            pool.join()
            
            converted_decoys = [m for m in converted_decoys if m is not None]
            print(f"  成功转换 {len(converted_decoys)} 个诱饵分子", flush=True)
            
            for m in converted_decoys:
                mol_data.append({
                    'atoms': m['atom_types'],
                    'coordinates': m['coords'],
                    'smi': m['smi'],
                    'mol': m['mol'],
                    'label': 0
                })
        
        # 4. 写入分子 LMDB

        if mol_data:
            mols_lmdb_path = os.path.join(target_output_dir, 'mols.lmdb')
            if os.path.exists(mols_lmdb_path):
                import shutil
                if os.path.isdir(mols_lmdb_path):
                    shutil.rmtree(mols_lmdb_path)
                else:
                    os.remove(mols_lmdb_path)
                print(f"  - 已删除现有的 LMDB 数据库: {mols_lmdb_path}", flush=True)
            write_lmdb(mol_data, mols_lmdb_path)
            print(f"\n目标 {target_name} 处理完成!", flush=True)
            print(f"  - 口袋文件: {pocket_lmdb_path}", flush=True)
            print(f"  - 分子文件: {mols_lmdb_path} (共 {len(mol_data)} 个分子)", flush=True)
        else:
            print(f"\n警告: 目标 {target_name} 没有有效的分子数据", flush=True)
        


def main():
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80, flush=True)
    print("步骤 1: 数据转换（生成 LMDB 文件）", flush=True)
    print("="*80, flush=True)
    
    # 处理输入数据并转换为 LMDB
    deal_input_data(args.input, args.dataset_name)
    
    print("\n" + "="*80, flush=True)
    print("数据转换完成!", flush=True)
    print("="*80, flush=True)
    
    print("\n" + "="*80, flush=True)
    print("步骤 2: 运行模型测试", flush=True)
    print("="*80, flush=True)
    
    # 调用 test.sh
    import subprocess
    import os
    
    script_dir = Path(__file__).parent
    test_sh_path = script_dir / "test.sh"
    
    if test_sh_path.exists():
        print(f"正在调用 test.sh: {test_sh_path}", flush=True)
        print(f"传递参数:", flush=True)
        print(f"  - results_path: {args.results_path}", flush=True)
        print(f"  - batch_size: {args.batch_size}", flush=True)
        print(f"  - checkpoint: {args.checkpoint}", flush=True)
        print(f"  - CUDA_VISIBLE_DEVICES: {args.gpu_id}", flush=True)
        
        try:
            # 设置环境变量传递参数给 test.sh
            print(6666, args.output, flush=True)
            env = os.environ.copy()
            env['RESULTS_PATH'] = str(args.output)
            env['BATCH_SIZE'] = str(args.batch_size)
            env['WEIGHT_PATH'] = str(args.checkpoint)+"/checkpoint_best.pt"
            # 移除 TASK 环境变量,现在使用通用推理
            # env['TASK'] = str(args.test_task)
            env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
            env['PYTHONUNBUFFERED'] = '1'  # 禁用 Python 缓冲
            
            print("="*80, flush=True)
            print("开始执行 test.sh，所有输出将实时显示：", flush=True)
            print("="*80, flush=True)
            
            # 不捕获输出，直接打印到终端
            result = subprocess.run(
                ["bash", str(test_sh_path)],
                cwd=str(script_dir),
                env=env,
                check=True
            )
            print("="*80, flush=True)
            print("test.sh 执行成功!", flush=True)
            print("="*80, flush=True)
        except subprocess.CalledProcessError as e:
            print("="*80, flush=True)
            print(f"test.sh 执行失败! 返回码: {e.returncode}", flush=True)
            print("="*80, flush=True)
            raise
    else:
        print(f"警告: test.sh 不存在于 {test_sh_path}", flush=True)
    
    # # 执行预测
        # result = predict(item, args.checkpoint, args.batch_size)
        
        # # 保存结果
        # save_output(output_dir, item['name'], result)
        # print(f"已保存结果: {item['name']}")


if __name__ == '__main__':
    main()