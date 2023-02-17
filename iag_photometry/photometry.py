from __future__ import annotations
from zipfile import ZipFile
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from photutils.aperture import SkyCircularAperture, aperture_photometry
from typing import Optional


class Photometry:
    def __init__(self, target_coord: SkyCoord):
        self._target_coord = target_coord
        self._photometry: Optional[pd.DataFrame] = None
        self._coords: Optional[SkyCoord] = None
        self._apertures: Optional[SkyCircularAperture] = None
        self._columns = []

    @staticmethod
    def process_zipfile(filename: str, target_coord: SkyCoord) -> Photometry:
        # create Photometry
        phot = Photometry(target_coord)

        # open zip file
        with ZipFile(filename) as zip:
            # loop all files
            for filename in zip.namelist():
                # open FITS file
                with zip.open(filename, "r") as f:
                    # open FITS file and get CCDData
                    fd = fits.open(f)
                    phot(fd)
                    fd.close()

        # finished
        return phot

    def __call__(self, hdu_list: fits.HDUList):
        # get CCDData
        data = CCDData(data=hdu_list[1].data, header=hdu_list[1].header, unit=u.adu, wcs=WCS(hdu_list[1].header))

        # first image?
        if self._photometry is None:
            self._process_first_image(hdu_list)

        # do photometry
        phot_table = aperture_photometry(data, self._apertures).to_pandas()
        phot_table.index = self._columns

        # define new row
        new_row = pd.DataFrame(phot_table["aperture_sum"]).T
        new_row.reset_index(drop=True, inplace=True)

        # add time
        new_row["time"] = [Time(data.header["DATE-OBS"]).jd]

        if self._photometry is None:
            self._photometry = new_row
        else:
            self._photometry = pd.concat([self._photometry, new_row], ignore_index=True)

    def _process_first_image(self, hdu_list: fits.HDUList):
        # read catalog
        cat = Table(hdu_list["CAT"].data).to_pandas()

        # build SkyCoords
        catalog = SkyCoord(ra=cat["ra"] * u.degree, dec=cat["dec"] * u.degree)
        idx, _, _ = self._target_coord.match_to_catalog_sky(catalog)
        target_row = cat.iloc[idx]

        # filter comparison stars
        cat.drop([idx], axis=0, inplace=True)
        cat = cat[(cat["peak"] > 10000) & (cat["peak"] < 40000)]
        cat = cat.append(target_row)

        # coords for aperture
        self._coords = SkyCoord(ra=cat["ra"] * u.degree, dec=cat["dec"] * u.degree)

        # apertures
        self._apertures = SkyCircularAperture(self._coords, r=4.0 * u.arcsec)

        # column names
        self._columns = [f"c{no}" for no in range(len(cat) - 1)] + ["t"]

    def photometry(self):
        # add all comps
        phot = self._photometry.copy()
        phot["c"] = phot[self._columns[:-1]].sum(axis=1)
        phot["rt"] = phot["t"] / phot["c"]
        return phot


__all__ = ["Photometry"]
