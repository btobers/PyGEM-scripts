import argparse
import os
import shutil
import pygem_input as pygem_prms


def main():
    parser = argparse.ArgumentParser(description="Script to make duplicate oggm glacier directories - primarily to avoid corruption if parellelizing runs on a single glacier")
    # add arguments
    parser.add_argument('-rgi_glac_number', type=str, default=None,
                        help='Randoph Glacier Inventory region')
    parser.add_argument('-num_copies', type=int, default=1,
                        help='Number of copies to create of the glacier directory data')
    args = parser.parse_args()
    num_copies = args.num_copies
    glac_num = args.rgi_glac_number

    if (glac_num is not None) and (num_copies)>1:
        reg,id = glac_num.split('.')
        reg = reg.zfill(2)
        thous = id[:2]
        
        root = pygem_prms.oggm_gdir_fp
        sfix = '/per_glacier/' + f'RGI60-{reg}/' + f'RGI60-{reg}.{thous}/'

        for n in range(num_copies):
            nroot = root.replace('gdirs',f'gdirs_{n+1}')
            # duplicate strucutre
            os.makedirs(nroot + sfix + f'RGI60-{reg}.{id}', exist_ok=True)
            # copy directory data
            shutil.copytree(root + sfix + f'RGI60-{reg}.{id}', nroot + sfix + f'RGI60-{reg}.{id}', dirs_exist_ok=True)

    return

if __name__ == '__main__':
    main()