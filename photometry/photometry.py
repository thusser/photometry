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
from photutils.aperture import SkyCircularAperture, aperture_photometry, SkyCircularAnnulus, ApertureStats
from typing import Optional, Tuple


class Photometry:
    def __init__(
        self,
        target_coord: SkyCoord,
        min_edge_dist: float = 20,
        aperture_radius: float = 4.0,
        sky_annulus_radii: Tuple[float, float] = (6.0, 8.0),
    ):
        self._target_coord = target_coord
        self._photometry: Optional[pd.DataFrame] = None
        self._coords: Optional[SkyCoord] = None
        self._apertures: Optional[SkyCircularAperture] = None
        self._sky_apertures: Optional[SkyCircularAnnulus] = None
        self._columns = []
        self._min_edge_dist = min_edge_dist
        self._aperture_radius = aperture_radius
        self._sky_annulus_radii = sky_annulus_radii

    @staticmethod
    def process_zipfile(filename: str, target_coord: SkyCoord, **kwargs) -> Photometry:
        # create Photometry
        phot = Photometry(target_coord, **kwargs)

        # open zip file
        with ZipFile(filename) as zip:
            # loop all files
            for filename in sorted(zip.namelist()):
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
        wcs = WCS(hdu_list[1].header)
        data = CCDData(data=hdu_list[1].data, header=hdu_list[1].header, unit=u.adu, wcs=wcs)

        # first image?
        if self._photometry is None:
            try:
                self._process_first_image(hdu_list)
            except KeyError as e:
                print(f"Could not process image as first image: {e}")
                return

        # do photometry
        phot_table = aperture_photometry(data, self._apertures).to_pandas()
        phot_table.index = self._columns

        # background
        aperstats = ApertureStats(data, self._sky_apertures)
        phot_table["background_mean"] = aperstats.mean.value
        phot_table["aperture_area"] = self._apertures.to_pixel(wcs).area_overlap(data)
        phot_table["background_total"] = phot_table["background_mean"] * phot_table["aperture_area"]
        phot_table["target-sky"] = phot_table["aperture_sum"] - phot_table["background_total"]

        # define new row
        new_row = pd.DataFrame(phot_table["target-sky"]).T
        new_row.reset_index(drop=True, inplace=True)

        # add time as DATE_OBS + EXPTIME/2
        new_row["time"] = [(Time(data.header["DATE-OBS"]) + (data.header["EXPTIME"] / 2.0) * u.second).jd]

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

        # drop target from catalog
        cat.drop([idx], axis=0, inplace=True)

        # filter comparison stars by peak value
        cat = cat[(cat["peak"] > 10000) & (cat["peak"] < 40000)]

        # filter comparison stars by edge distance
        height, width = hdu_list[1].data.shape
        m = self._min_edge_dist
        cat = cat[(cat["x"] > m) & (cat["y"] > m) & (cat["x"] < width - m) & (cat["y"] < height - m)]

        # re-add target to catalog
        cat = pd.concat([cat, pd.DataFrame(target_row).T])

        # coords for aperture
        self._coords = SkyCoord(ra=cat["ra"] * u.degree, dec=cat["dec"] * u.degree)

        # apertures
        self._apertures = SkyCircularAperture(self._coords, r=self._aperture_radius * u.arcsec)
        self._sky_apertures = SkyCircularAnnulus(
            self._coords, r_in=self._sky_annulus_radii[0] * u.arcsec, r_out=self._sky_annulus_radii[1] * u.arcsec
        )

        # column names
        self._columns = [f"c{no}" for no in range(len(cat) - 1)] + ["t"]

    def photometry(self):
        # nothing?
        if self._photometry is None:
            return pd.DataFrame()

        # add all comps
        phot = self._photometry.copy()
        phot["c"] = phot[self._columns[:-1]].sum(axis=1)
        phot["rt"] = phot["t"] / phot["c"]
        return phot


__all__ = ["Photometry"]
