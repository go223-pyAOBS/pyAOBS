/*--------------------------------------------------------------------
 *    The GMT-system:	@(#)gmt_project.h	3.4  09 Aug 1995
 *
 *    Copyright (c) 1991-1995 by P. Wessel and W. H. F. Smith
 *    See README file for copying and redistribution conditions.
 *--------------------------------------------------------------------*/
/*
 * Include file for programs that use the map-projections functions.  Note
 * that most programs will include this by including gmt.h
 *
 * Author:	Paul Wessel
 * Date:	26-FEB-1990
 *
 */


#define D2R (M_PI / 180.0)
#define R2D (180.0 / M_PI)
#define d_sqrt(x) ((x) < 0.0 ? 0.0 : sqrt (x))

typedef int BOOLEAN;		/* BOOLEAN used for logical variables */


struct MAP_PROJECTIONS {

	double pars[10];		/* Raw unprocessed map-projection parameters as passed on command line */
	double z_pars[2];		/* Raw unprocessed z-projection parameters as passed on command line */
	
	/* Common projection parameters */
	
	int projection;			/* Gives the id number for the projection used */
	
	BOOLEAN x_off_supplied;		/* Used to set xorigin/yorigin for overlay/final plots */
	BOOLEAN y_off_supplied;
	BOOLEAN region_supplied;	/* TRUE when -R option has been given */
	BOOLEAN units_pr_degree;	/* TRUE if scale is given as inch (or cm)/degree.  FALSE for 1:xxxxx */
	BOOLEAN gave_map_width;		/* TRUE if map width in inch (or cm) is given instead of scale.  FALSE for 1:xxxxx */
	BOOLEAN region;			/* TRUE if -R gives lon/lat boundaires, FALSE if it gives lower/left upper right corners */
	BOOLEAN north_pole;		/* TRUE if projection is on northern hermisphere, FALSE on southern */
	BOOLEAN edge[4];		/* TRUE if the edge is a map boundary */
	BOOLEAN three_D;		/* Parameters for 3-D projections */

	double x0, y0, z0;		/* Projected values of the logical origin for the projection */
	double xmin, xmax, ymin, ymax, zmin, zmax;	/* Extreme projected values */
	double w, e, s, n;		/* Bounding geographical region, if applicable */
	double z_bottom, z_top;		/* Bounds in z direction */
	double x_scale, y_scale;	/* Scaling for meters to map-distance (typically inch) conversion */
	double z_scale;			/* Scale for z projection (user-units to map-units (typically inch) */
	double z_level;			/* Level at which to draw basemap [0] */
	double unit;			/* Gives meters pr plot unit (0.01 or 0.0254) */
	double central_meridian;	/* Central meridian for projection */
	double pole;			/* +90 pr -90, depending on which pole */
	
	
	/* TM and UTM Projections */
	
	double t_e2;
	double t_c1, t_c2, t_c3, t_c4;
	double t_ic1, t_ic2, t_ic3, t_ic4;
	
	/* Lambert Azimuthal Equal-Area Projection */
	
	double sinp;
	double cosp;
	
	/* Stereographic Projection */
	
	double s_cosz1, s_sinz1;
	double s_c;
	double r;		/* Radius of projected sphere in plot units (inch or cm) */
	BOOLEAN polar;		/* True if projection pole coincides with S or N pole */
	
	
};

