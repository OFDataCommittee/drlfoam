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
    Foam::agentRotatingWallVelocityFvPatchVectorField

Group
    grpWallBoundaryConditions grpGenericBoundaryConditions

Description
    This boundary condition provides a rotational velocity condition.

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
        type            agentRotatingWallVelocity;
        origin          (0 0 0);
        axis            (0 0 1);
        policy          "policy.pt"
        seed            0;
        train           true;
        absOmegaMax     0.05;
    }
    \endverbatim

   
See also
    Foam::fixedValueFvPatchField

SourceFiles
    agentRotatingWallVelocityFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef agentRotatingWallVelocityFvPatchVectorField_H
#define agentRotatingWallVelocityFvPatchVectorField_H

#include <torch/script.h>
#include <random>
#include "fixedValueFvPatchFields.H"
#include "probes.H"
#include "timeControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
           Class agentRotatingWallVelocityFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class agentRotatingWallVelocityFvPatchVectorField
:
    public fixedValueFvPatchVectorField
{
    // Private data

        //- origin of the rotation
        vector origin_;

        //- axis of the rotation
        vector axis_;

        //- training or evaluation mode
        bool train_;

        //- name of the PyTorch angular velocity model
        word policy_name_;

        //- PyTorch model of angular velocity
        torch::jit::script::Module policy_;

        //- largest allowed absolute value of angular velocity
        scalar abs_omega_max_;

        //- seed for random sampling
        int seed_;

        //- name of the probes function object dictionary
        word probes_name_;

        //- random number generator
        std::mt19937 gen_;

        //- next predicted angular velocity
        scalar omega_;

        //- previously predicted angular velocity
        scalar omega_old_;

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

public:

    //- Runtime type information
    TypeName("agentRotatingWallVelocity");


    // Constructors

        //- Construct from patch and internal field
        agentRotatingWallVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        agentRotatingWallVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given agentRotatingWallVelocityFvPatchVectorField
        //  onto a new patch
        agentRotatingWallVelocityFvPatchVectorField
        (
            const agentRotatingWallVelocityFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        agentRotatingWallVelocityFvPatchVectorField
        (
            const agentRotatingWallVelocityFvPatchVectorField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchVectorField> clone() const
        {
            return tmp<fvPatchVectorField>
            (
                new agentRotatingWallVelocityFvPatchVectorField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        agentRotatingWallVelocityFvPatchVectorField
        (
            const agentRotatingWallVelocityFvPatchVectorField&,
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
                new agentRotatingWallVelocityFvPatchVectorField(*this, iF)
            );
        }



    // Member functions

        //- Update the coefficients associated with the patch field
        virtual void updateCoeffs();

        //- Write
        virtual void write(Ostream&) const;

        //- Save trajectory to file
        void saveTrajectory(scalar alpha, scalar beta) const;

        //- Get probes function object dictionary
        const Foam::dictionary& getProbesDict();

        //- initialize probes function object
        Foam::probes initializeProbes();

        //- initialize control
        Foam::timeControl initializeControl();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
