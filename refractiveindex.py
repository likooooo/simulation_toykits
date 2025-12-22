import os
import yaml
import sys
import numpy as np
import scipy.interpolate
from io import open

try:
    from yaml import CBaseLoader as BaseLoader
except ImportError:
    from yaml import BaseLoader

# import collections


# Commit on January 3, 2025 just prior to a major update of the database structure.
# List of commits here: https://github.com/polyanskiy/refractiveindex.info-database/commits/master/
_DATABASE_SHA = "451b9136b4b3566f6259b703990add5440ca125f"

_MASTER_URL = f'https://github.com/polyanskiy/refractiveindex.info-database/archive/{_DATABASE_SHA}.zip'


class RefractiveIndex:
    """Class that parses the refractiveindex.info YAML database"""

    def __init__(self, 
                 databasePath=os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database"), 
                 auto_download : bool = True,
                 ssl_certificate_location: str = None,
                 update_database : bool = False,
                 sorted_by_year : bool = False):
        """
        Initializes the RefractiveIndex class by downloading and parsing the refractiveindex.info YAML database.

        Args:
            databasePath (str): The path where the database will be stored. Defaults to ~/.refractiveindex.info-database.
            auto_download (bool): Whether to automatically download the database if it doesn't exist. Defaults to True.
            update_database (bool): If True the database is being downloaded again, even if it already exists. Defaults to False.
            ssl_certificate_location (str | None): The path to a custom SSL certificate file to use for verification. If None, the default SSL context will be used. If an empty string, SSL verification will be disabled.

        Raises:
            Exception: If the database cannot be downloaded or extracted.

        Notes:
            If auto_download is True and the database does not exist, the script will download the latest version of the database from GitHub and extract it to the specified path.
        """

        if not os.path.exists(databasePath) and auto_download or update_database:
          import tempfile, urllib.request, zipfile, shutil, ssl
          with tempfile.TemporaryDirectory() as tempdir:
            zip_filename = os.path.join(tempdir, "db.zip")
            print("downloading refractiveindex.info database...", file=sys.stderr)
            
            # ignore ssl certificate errors
            if ssl_certificate_location is not None :
                if ssl_certificate_location == "" :
                    # have the option to disable ssl verification
                    ssl._create_default_https_context = ssl._create_unverified_context
                else : 
                    # Create an SSL context with your CA bundle
                    if  ssl_certificate_location[-4:] != ".pem" or not os.path.isfile(ssl_certificate_location):
                        raise ValueError(f"It does not appear that the path you supplied points to an existing certificate file with '.pem' at the end: {ssl_certificate_location}")
                    ssl._create_default_https_context = ssl.create_default_context(cafile=ssl_certificate_location)
                
            urllib.request.urlretrieve(_MASTER_URL, zip_filename)
            print("extracting...", file=sys.stderr)
            with zipfile.ZipFile(zip_filename, 'r') as zf: zf.extractall(tempdir)
            
            # remove the old database if it exists. We should only arrive here if update_database = True or it doesn't exist
            if os.path.isdir(databasePath) : 
                print("Removing the old database...")
                shutil.rmtree(databasePath)
            
            # move the downloaded database to the database location
            shutil.move(os.path.join(tempdir, f"refractiveindex.info-database-{_DATABASE_SHA}", "database"), databasePath)
            print("done", file=sys.stderr)

        self.referencePath = os.path.normpath(databasePath)
        fileName = os.path.join(self.referencePath, "catalog-nk.yml" + ["", ".sorted"][int(sorted_by_year)])
        with open(fileName, "rt", encoding="utf-8") as f:
            self.catalog = yaml.load(f, Loader=BaseLoader)

        # TODO: Do i NEED namedtuples, or am i just wasting time?
        # Shelf = collections.namedtuple('Shelf', ['SHELF', 'name', 'books'])
        # Book = collections.namedtuple('Book', ['BOOK', 'name', 'pages'])
        # Page = collections.namedtuple('Page', ['PAGE', 'name', 'path'])

        # self.catalog = [Shelf(**shelf) for shelf in rawCatalog]
        # for shelf in self.catalog:
        #     books = []
        #     for book in shelf.books:
        #         rawBook = book
        #         if not 'divider' in rawBook:
        #             books.append(Book(**rawBook))
        #         pages = []
        #         for page in book.pages:
        #             rawPage = page
        #             pages.append(Page(**rawPage))
        #         book.pages = pages

    def getMaterialFilename(self, book, page=None, shelf=None):
        """
        通过遍历目录查找并返回材料文件的完整路径。
        如果 page 或 shelf 未指定或不存在，将尝试自动选择第一个可用的选项并打印警告。

        :param shelf: 所需材料所在的“架子”（顶层类别）。如果为 None 或找不到，则自动选择第一个。
        :param book: 所需材料的“书名”（例如材料名称）。
        :param page: 所需材料的“页码”（例如数据源名称）。如果为 None 或找不到，则自动选择第一个。
        :return: 找到的材料文件的完整路径字符串。
        :raises ValueError: 如果找不到指定的 book 或 catalog 为空，则抛出此异常。
        """
        
        # --- 1. 确定要遍历的 Shelf 列表 ---
        shelf_list_to_check = []
        
        if shelf is not None:
            # 如果指定了 shelf，只检查这一个
            shelf_list_to_check = [s for s in self.catalog if s.get('SHELF') == shelf]
        
        if not shelf_list_to_check:
            # 如果 shelf 为 None 或指定 shelf 没找到，则使用整个 catalog
            shelf_list_to_check = self.catalog
            if shelf is not None:
                # 打印警告：指定 shelf 找不到，将使用第一个
                print(f"Warning: Specified shelf '{shelf}' not found. Searching all available shelves.", file=sys.stderr)
            elif shelf is None and self.catalog:
                # 打印警告：shelf 未指定，将使用第一个可用的 shelf
                first_shelf_name = self.catalog[0].get('SHELF', 'N/A')
                print(f"Warning: Shelf not specified. Using the first available shelf: '{first_shelf_name}'.", file=sys.stderr)

        # --- 2. 遍历 Shelf, Book, 和 Page ---
        
        # 为了简化逻辑，我们主要遍历 shelf_list_to_check 并寻找 book
        for sh in shelf_list_to_check:
            current_shelf_name = sh.get('SHELF')
            
            # 遍历 BOOK
            for b in sh.get('content', []):
                if 'DIVIDER' not in b:
                    if b.get('BOOK') == book:
                        # 找到了 BOOK，现在开始处理 PAGE

                        # 收集所有非 DIVIDER 的 Page 选项
                        page_options = [p for p in b.get('content', []) if 'DIVIDER' not in p]
                        
                        if not page_options:
                            # 找到了 Book，但它下面没有 Page 数据，跳过当前 Book/Shelf 组合
                            continue 

                        # 尝试查找指定的 Page
                        matching_page = None
                        if page is not None:
                            for p in page_options:
                                if p.get('PAGE') == page:
                                    matching_page = p
                                    break

                        # 如果未找到指定的 Page 或 page=None，则使用第一个 Page
                        if matching_page is None:
                            matching_page = page_options[0] # 使用第一个非分隔符 Page
                            if page is not None:
                                print(f"Warning: Specified page '{page}' not found for book '{book}'. Using first available page: '{matching_page.get('PAGE')}'.", file=sys.stderr)
                            elif page is None:
                                print(f"Warning: Page not specified for book '{book}'. Using first available page: '{matching_page.get('PAGE')}'.", file=sys.stderr)
                        return os.path.join(self.referencePath, 'data-nk', os.path.normpath(matching_page['data']))
        raise ValueError(f"Error: Material/Book '{book}' not found in the entire catalog.")


    def getMaterial(self, book, page=None, shelf='main'):
        """

        :param shelf:
        :param book:
        :param page:
        :return:
        """
        return Material(self.getMaterialFilename(shelf=shelf, book=book, page=page))


class Material:
    """ Material class"""

    def __init__(self, filename):
        """

        :param filename:
        """
        self.refractiveIndex = None
        self.extinctionCoefficient = None
        self.originalData = None

        with open(filename, "rt", encoding="utf-8") as f:
            material = yaml.load(f, Loader=BaseLoader)

        for data in material['DATA']:
            if (data['type'].split())[0] == 'tabulated':
                rows = data['data'].split('\n')
                splitrows = [c.split() for c in rows]
                wavelengths = []
                n = []
                k = []
                for s in splitrows:
                    if len(s) > 0:
                        wavelengths.append(float(s[0]))
                        n.append(float(s[1]))
                        if len(s) > 2:
                            k.append(float(s[2]))

                if (data['type'].split())[1] == 'n':

                    if self.refractiveIndex is not None:
                        Exception('Bad Material YAML File')

                    self.refractiveIndex = RefractiveIndexData.setupRefractiveIndex(formula=-1,
                                                                                    wavelengths=wavelengths,
                                                                                    values=n)
                    self.originalData = {'wavelength (um)': np.array(wavelengths), 
                                         'n' : np.array(n)}
                elif (data['type'].split())[1] == 'k':

                    self.extinctionCoefficient = ExtinctionCoefficientData.setupExtinctionCoefficient(wavelengths, n)
                    self.originalData = {'wavelength (um)': np.array(wavelengths), 
                                         'n' : 1j*np.array(n)}

                elif (data['type'].split())[1] == 'nk':

                    if self.refractiveIndex is not None:
                        Exception('Bad Material YAML File')

                    self.refractiveIndex = RefractiveIndexData.setupRefractiveIndex(formula=-1,
                                                                                    wavelengths=wavelengths,
                                                                                    values=n)
                    self.extinctionCoefficient = ExtinctionCoefficientData.setupExtinctionCoefficient(wavelengths, k)
                    self.originalData = {'wavelength (um)': np.array(wavelengths), 
                                         'n' : np.array(n) + 1j*np.array(k)}
                    
            elif (data['type'].split())[0] == 'formula':

                if self.refractiveIndex is not None:
                    Exception('Bad Material YAML File')

                formula = int((data['type'].split())[1])
                coefficents = [float(s) for s in data['coefficients'].split()]
                for k in ['range','wavelength_range']:
                    if k in data:
                        break
                rangeMin = float(data[k].split()[0])
                rangeMax = float(data[k].split()[1])

                self.refractiveIndex = RefractiveIndexData.setupRefractiveIndex(formula=formula,
                                                                                rangeMin=rangeMin,
                                                                                rangeMax=rangeMax,
                                                                                coefficients=coefficents)
                wavelengths = np.linspace(rangeMin, rangeMax, 1000)
                self.originalData = {'wavelength (um)': wavelengths,
                                     'n' : self.refractiveIndex.getRefractiveIndex(wavelength=wavelengths*1000)}
                

    def getRefractiveIndex(self, wavelength, bounds_error=True):
        """

        :param wavelength:
        :return: :raise Exception:
        """
        if self.refractiveIndex is None:
            raise Exception('No refractive index specified for this material')
        else:
            return self.refractiveIndex.getRefractiveIndex(wavelength, bounds_error=bounds_error)

    def getExtinctionCoefficient(self, wavelength, bounds_error=True):
        """

        :param wavelength:
        :return: :raise NoExtinctionCoefficient:
        """
        if self.extinctionCoefficient is None:
            raise NoExtinctionCoefficient('No extinction coefficient specified for this material')
        else:
            return self.extinctionCoefficient.getExtinctionCoefficient(wavelength, bounds_error=bounds_error)


#
# Refractive Index
#
class RefractiveIndexData:
    """Abstract RefractiveIndex class"""

    @staticmethod
    def setupRefractiveIndex(formula, **kwargs):
        """

        :param formula:
        :param kwargs:
        :return: :raise Exception:
        """
        if formula >= 0:
            return FormulaRefractiveIndexData(formula, **kwargs)
        elif formula == -1:
            return TabulatedRefractiveIndexData(**kwargs)
        else:
            raise Exception('Bad RefractiveIndex data type')

    def getRefractiveIndex(self, wavelength):
        """

        :param wavelength:
        :raise NotImplementedError:
        """
        raise NotImplementedError('Different for functionally and experimentally defined materials')


class FormulaRefractiveIndexData:
    """Formula RefractiveIndex class"""

    def __init__(self, formula, rangeMin, rangeMax, coefficients):
        """

        :param formula:
        :param rangeMin:
        :param rangeMax:
        :param coefficients:
        """
        self.formula = formula
        self.rangeMin = rangeMin
        self.rangeMax = rangeMax
        self.coefficients = coefficients

    def getRefractiveIndex(self, wavelength, bounds_error=True):
        """

        :param wavelength:
        :return: :raise Exception:
        """
        wavelength = np.copy(wavelength)/1000.0
        if self.rangeMin <= np.min(wavelength) <= self.rangeMax and self.rangeMin <= np.max(wavelength) <= self.rangeMax or not bounds_error:
            formula_type = self.formula
            coefficients = self.coefficients
            n = 0
            if formula_type == 1:  # Sellmeier
                nsq = 1 + coefficients[0]
                g = lambda c1, c2, w: c1 * (w ** 2) / (w ** 2 - c2 ** 2)
                for i in range(1, len(coefficients), 2):
                    nsq += g(coefficients[i], coefficients[i + 1], wavelength)
                n = np.sqrt(nsq)
            elif formula_type == 2:  # Sellmeier-2
                nsq = 1 + coefficients[0]
                g = lambda c1, c2, w: c1 * (w ** 2) / (w ** 2 - c2)
                for i in range(1, len(coefficients), 2):
                    nsq += g(coefficients[i], coefficients[i + 1], wavelength)
                n = np.sqrt(nsq)
            elif formula_type == 3:  # Polynomal
                g = lambda c1, c2, w: c1 * w ** c2
                nsq = coefficients[0]
                for i in range(1, len(coefficients), 2):
                    nsq += g(coefficients[i], coefficients[i + 1], wavelength)
                n = np.sqrt(nsq)
            elif formula_type == 4:  # RefractiveIndex.INFO
                g1 = lambda c1, c2, c3, c4, w: c1 * w**c2 / (w**2 - c3**c4)
                g2 = lambda c1, c2, w: c1 * w**c2
                nsq = coefficients[0]
                for i in range(1, min(8, len(coefficients)), 4):
                    nsq += g1(coefficients[i], coefficients[i+1], coefficients[i+2], coefficients[i+3], wavelength)
                if len(coefficients) > 9:
                    for i in range(9, len(coefficients), 2):
                        nsq += g2(coefficients[i], coefficients[i+1], wavelength)
                n = np.sqrt(nsq)
            elif formula_type == 5:  # Cauchy
                g = lambda c1, c2, w: c1 * w ** c2
                n = coefficients[0]
                for i in range(1, len(coefficients), 2):
                    n += g(coefficients[i], coefficients[i + 1], wavelength)
            elif formula_type == 6:  # Gasses
                n = 1 + coefficients[0]
                g = lambda c1, c2, w: c1 / (c2 - w ** (-2))
                for i in range(1, len(coefficients), 2):
                    n += g(coefficients[i], coefficients[i + 1], wavelength)
            elif formula_type == 7:  # Herzberger
                g1 = lambda c1, w, p: c1 / (w**2 - 0.028)**p
                g2 = lambda c1, w, p: c1 * w**p
                n = coefficients[0]
                n += g1(coefficients[1], wavelength, 1)
                n += g1(coefficients[2], wavelength, 2)
                for i in range(3, len(coefficients)):
                    n += g2(coefficients[i], wavelength, 2*(i-2))
            elif formula_type == 8:  # Retro
                raise FormulaNotImplemented('Retro formula not yet implemented')
            elif formula_type == 9:  # Exotic
                raise FormulaNotImplemented('Exotic formula not yet implemented')
            else:
                raise Exception('Bad formula type')

            n = np.where((self.rangeMin<=wavelength) & (wavelength<=self.rangeMax), n, np.nan)
            return n
        else:
            raise Exception(
                'Wavelength {} um is out of bounds. Correct range: ({} um, {} um)'.format(wavelength, self.rangeMin,
                                                                                     self.rangeMax))


class TabulatedRefractiveIndexData:
    """Tabulated RefractiveIndex class"""

    def __init__(self, wavelengths, values):
        """

        :param wavelengths:
        :param values:
        """
        self.rangeMin = np.min(wavelengths)
        self.rangeMax = np.max(wavelengths)

        if self.rangeMin == self.rangeMax:
            self.refractiveFunction = values[0]
        else:
            self.refractiveFunction = scipy.interpolate.interp1d(wavelengths, values, bounds_error=False)

    def getRefractiveIndex(self, wavelength, bounds_error=True):
        """

        :param wavelength:
        :return: :raise Exception:
        """
        wavelength = np.copy(wavelength)/1000.0
        if self.rangeMin == self.rangeMax and self.rangeMin == wavelength:
            return self.refractiveFunction
        elif self.rangeMin <= np.min(wavelength) <= self.rangeMax and self.rangeMin <= np.max(wavelength) <= self.rangeMax and self.rangeMin != self.rangeMax or not bounds_error:
            return self.refractiveFunction(wavelength)
        else:
            raise Exception(
                'Wavelength {} um is out of bounds. Correct range: ({} um, {} um)'.format(wavelength, self.rangeMin,
                                                                                     self.rangeMax))


#
# Extinction Coefficient
#
class ExtinctionCoefficientData:
    """ExtinctionCofficient class"""

    @staticmethod
    def setupExtinctionCoefficient(wavelengths, values):
        """

        :param wavelengths:
        :param values:
        :return:
        """
        return ExtinctionCoefficientData(wavelengths, values)

    def __init__(self, wavelengths, coefficients):
        """

        :param wavelengths:
        :param coefficients:
        """
        self.extCoeffFunction = scipy.interpolate.interp1d(wavelengths, coefficients, bounds_error=False)
        self.rangeMin = np.min(wavelengths)
        self.rangeMax = np.max(wavelengths)

    def getExtinctionCoefficient(self, wavelength, bounds_error=True):
        """

        :param wavelength:
        :return: :raise Exception:
        """
        wavelength = np.copy(wavelength)/1000.0
        if self.rangeMin <= np.min(wavelength) <= self.rangeMax and self.rangeMin <= np.max(wavelength) <= self.rangeMax or not bounds_error:
            return self.extCoeffFunction(wavelength)
        else:
            raise Exception(
                'Wavelength {} um is out of bounds. Correct range: ({} um, {} um)'.format(wavelength, self.rangeMin,
                                                                                     self.rangeMax))


#
# Custom Exceptions
#
class FormulaNotImplemented(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class NoExtinctionCoefficient(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
    

class RefractiveIndexMaterial:
    def __init__(self, shelf, book, page, **ri_kwargs):
        BD = RefractiveIndex(**ri_kwargs)
        self.material = BD.getMaterial(shelf=shelf, book=book, page=page)
        
    def get_refractive_index(self, wavelength_nm):
        return self.material.getRefractiveIndex(np.copy(wavelength_nm))
    
    def get_extinction_coefficient(self, wavelength_nm):
        return self.material.getExtinctionCoefficient(np.copy(wavelength_nm))
    
    def get_epsilon(self, wavelength_nm, exp_type='exp_minus_i_omega_t'):
        n = self.get_refractive_index(wavelength_nm)
        k = self.get_extinction_coefficient(wavelength_nm)
        if exp_type=='exp_minus_i_omega_t':
            return (n + 1j*k)**2
        else:
            return (n - 1j*k)**2
        
import streamlit as st
meterial_db_path = os.path.abspath("./assets/refractiveindex.info-database")
@st.cache_data
def load_refractive_index():
    return RefractiveIndex(databasePath=meterial_db_path)

@st.cache_data
def load_nk(shelf, book, page):
    if "Vacuum" == book:
        return np.array([0, 100]), np.array([1, 1]), np.array([0, 0]) 
    DB = load_refractive_index()
    material = DB.getMaterial(shelf=shelf, book=book, page=page)
    wl = material.originalData['wavelength (um)']

    wl_nm = wl * 1e3
    n, k = None, None
    try:
        n = material.getRefractiveIndex(wl_nm)
    except Exception:
        n = np.ones_like(wl) 
    try:
        k = material.getExtinctionCoefficient(wl_nm)
    except Exception:
        k = np.zeros_like(wl)
    if k is None: k = np.zeros_like(n)
    return wl, n, k
