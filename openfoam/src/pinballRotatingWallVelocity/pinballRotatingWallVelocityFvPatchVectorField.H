/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::pinballRotatingWallVelocityFvPatchVectorField

Group
    grpWallBoundaryConditions grpGenericBoundaryConditions

Description
    This boundary condition provides a rotational velocity condition.
    The boundary condition was originally implemented by Tom Krogmann
    during his student project at TU Braunschweig. For more information
    refer to:
        GitHub: https://github.com/TomKrogmann/Optimal_Sensor_Placement_for_Active_Flow_Control_in_Deep_Reinforcement_Learning
        report: https://zenodo.org/record/7636959#.Y-qTLcbMJkg

Usage
    \table
        Property     | Description             | Required    | Default value
        origin       | origin of rotation in Cartesian coordinates | yes|
        axis         | axis of rotation        | yes         |
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            pinballRotatingWallVelocity;
        origin_a        (1 0 0);
        origin_b        (0 1 0);
        origin_c        (0 -1 0);
        axis            (0 0 1);
        policy          "policy.pt"
        seed            0;
        train           true;
        absOmegaMax     5.0;
    }
    \endverbatim

See also
    Foam::fixedValueFvPatchField

SourceFiles
    pinballRotatingWallVelocityFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef pinballRotatingWallVelocityFvPatchVectorField_H
#define pinballRotatingWallVelocityFvPatchVectorField_H

#include <torch/script.h>
#include <random>
#include "fixedValueFvPatchFields.H"
#include "probes.H"
#include "timeControl.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
           Class pinballRotatingWallVelocityFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class pinballRotatingWallVelocityFvPatchVectorField
:
    public fixedValueFvPatchVectorField
{
    // Private data

        //- origin of cylinder a
        vector origin_a_;

        //- origin of cylinder b
        vector origin_b_;

        //- origin of cylinder c
        vector origin_c_;

        //- axis of the rotation
        vector axis_;

        //- training or evaluation mode
        bool train_;

        //- name of the PyTorch angular velocity model
        word policy_name_;

        //- PyTorch model predicting mean and log(std) of angular velocity
        torch::jit::script::Module policy_;

        //- largest allowed absolute value of angular velocity
        scalar abs_omega_max_;

        //- seed for random sampling
        int seed_;

        //- name of the probes function object dictionary
        word probes_name_;

        //- random number generator
        std::mt19937 gen_;

        //- next predicted angular velocity of cylinder a
        scalar omega_a_;

        //- previously predicted angular velocity of cylinder a
        scalar omega_old_a_;

        //- next predicted angular velocity of cylinder b
        scalar omega_b_;

        //- previously predicted angular velocity of cylinder b
        scalar omega_old_b_;

        //- next predicted angular velocity of cylinder c
        scalar omega_c_;

        //- previously predicted angular velocity of cylinder c
        scalar omega_old_c_;

        //- last time at which angular velocity was computed
        scalar control_time_;

        //- update of next angular velocity value if true
        bool update_omega_;

        //- when to start controlling
        scalar start_time_;

        //- time increment between control actions
        scalar dt_control_;

        //- probes function object
        Foam::probes probes_;

        //- timeControls consistent with function objects
        Foam::timeControl control_;

        //- lists of face centers, normals, and face IDs
        DynamicList <vector> centers_a_;
        DynamicList <vector> centers_b_;
        DynamicList <vector> centers_c_;
        DynamicList <vector> normals_a_;
        DynamicList <vector> normals_b_;
        DynamicList <vector> normals_c_;
        DynamicList <label> faces_a_;
        DynamicList <label> faces_b_;
        DynamicList <label> faces_c_;

public:

    //- Runtime type information
    TypeName("pinballRotatingWallVelocity");


    // Constructors

        //- Construct from patch and internal field
        pinballRotatingWallVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        pinballRotatingWallVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given pinballRotatingWallVelocityFvPatchVectorField
        //  onto a new patch
        pinballRotatingWallVelocityFvPatchVectorField
        (
            const pinballRotatingWallVelocityFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        pinballRotatingWallVelocityFvPatchVectorField
        (
            const pinballRotatingWallVelocityFvPatchVectorField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchVectorField> clone() const
        {
            return tmp<fvPatchVectorField>
            (
                new pinballRotatingWallVelocityFvPatchVectorField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        pinballRotatingWallVelocityFvPatchVectorField
        (
            const pinballRotatingWallVelocityFvPatchVectorField&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchVectorField> clone
        (
             const DimensionedField<vector, volMesh>& iF
        ) const
        {
            return tmp<fvPatchVectorField>
            (
                new pinballRotatingWallVelocityFvPatchVectorField(*this, iF)
            );
        }



    // Member functions

        //- Update the coefficients associated with the patch field
        virtual void updateCoeffs();

        //- Write
        virtual void write(Ostream&) const;

        //- Save trajectory to file
        void saveTrajectory(scalar alpha_a, scalar beta_a, scalar alpha_b, scalar beta_b, scalar alpha_c, scalar beta_c) const;

        //- Get probes function object dictionary
        const Foam::dictionary& getProbesDict();

        //- initialize probes function object
        Foam::probes initializeProbes();

        //- initialize control
        Foam::timeControl initializeControl();

        //- determine attribution of patch faces to cylinders a/b/c
        void initializeCylinderSegmentation();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
