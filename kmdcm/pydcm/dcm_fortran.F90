!///////////////////////////////////////////////////////////////////////////////
!
!   dcm_fortran.F90
!   Created: 27 April 2022 at 13:12
!   Author: Kai Töpfer
!   Based on MDCM code main.F90
!   Orignal author: Oliver Unke
!   Updated and extended by Mike Devereux
!
!///////////////////////////////////////////////////////////////////////////////
module dcm_fortran

    implicit none

    ! Data type
    integer, parameter :: rp = selected_real_kind(15)   ! python float

    ! Parameter
    real(rp), parameter :: a02ang   = 0.52917721067_rp
    real(rp), parameter :: ang2a0   = 1._rp/a02ang
    real(rp), parameter :: ha2kcal  = 627.50960803_rp

    ! Maximum MDCM charge offset
    real(rp), parameter :: maxlcl   = 0.5_rp/a02ang

    ! Numeric zero threshold - 1e-9
    real(rp), parameter :: eps  = 0.000000001_rp

    ! Van-der-Waals Radii (!https://de.wikipedia.org/wiki/Van-der-Waals-Radius)
    ! r must be smaller than scaling*vdw_radius to be feasible
    real(rp), parameter :: vdw_scaling = 1.0_rp/3.0_rp
    real(rp), dimension(55), parameter :: vdw_radius = (/ &
        1.20_rp*ang2a0, & ! H
        1.40_rp*ang2a0, & ! He
        1.82_rp*ang2a0, & ! Li
        1.53_rp*ang2a0, & ! Be
        1.92_rp*ang2a0, & ! B
        1.70_rp*ang2a0, & ! C
        1.55_rp*ang2a0, & ! N
        1.52_rp*ang2a0, & ! O
        1.47_rp*ang2a0, & ! F
        1.54_rp*ang2a0, & ! Ne
        2.27_rp*ang2a0, & ! Na
        1.73_rp*ang2a0, & ! Mg
        1.84_rp*ang2a0, & ! Al
        2.10_rp*ang2a0, & ! Si
        1.80_rp*ang2a0, & ! P
        1.80_rp*ang2a0, & ! S
        1.75_rp*ang2a0, & ! Cl
        1.88_rp*ang2a0, & ! Ar
        2.75_rp*ang2a0, & ! K
        2.31_rp*ang2a0, & ! Ca
        0.00_rp*ang2a0, & ! Sc
        0.00_rp*ang2a0, & ! Ti
        0.00_rp*ang2a0, & ! V
        0.00_rp*ang2a0, & ! Cr
        0.00_rp*ang2a0, & ! Mn
        0.00_rp*ang2a0, & ! Fe
        0.00_rp*ang2a0, & ! Co
        1.63_rp*ang2a0, & ! Ni
        1.40_rp*ang2a0, & ! Cu
        1.39_rp*ang2a0, & ! Zn
        1.87_rp*ang2a0, & ! Ga
        2.11_rp*ang2a0, & ! Ge
        1.85_rp*ang2a0, & ! As
        1.90_rp*ang2a0, & ! Se
        1.85_rp*ang2a0, & ! Br
        2.02_rp*ang2a0, & ! Kr
        3.03_rp*ang2a0, & ! Rb
        2.49_rp*ang2a0, & ! Sr
        0.00_rp*ang2a0, & ! Y
        0.00_rp*ang2a0, & ! Zr
        0.00_rp*ang2a0, & ! Nb
        0.00_rp*ang2a0, & ! Mo
        0.00_rp*ang2a0, & ! Tc
        0.00_rp*ang2a0, & ! Ru
        0.00_rp*ang2a0, & ! Rh
        1.63_rp*ang2a0, & ! Pd
        1.72_rp*ang2a0, & ! Ag
        1.58_rp*ang2a0, & ! Cd
        1.93_rp*ang2a0, & ! In
        2.17_rp*ang2a0, & ! Sn
        2.06_rp*ang2a0, & ! Sb
        2.06_rp*ang2a0, & ! Te
        1.98_rp*ang2a0, & ! I
        2.16_rp*ang2a0, & ! Xe
        0.00_rp*ang2a0  &
        /)

    ! Program parameters
    logical :: verbose = .true. ! Toggles printing mode

    ! Input data files information
    integer :: Nload
    character(len=1024), dimension(:), allocatable :: files_load_cube
    character(len=1024) :: file_load_cxyz, file_load_clcl, file_load_poly

    ! System information
    integer :: Natom    ! Number of system atoms
    integer :: Ngrid    ! Number of total grid points
    integer :: Nimpt     ! Number of important grid points
    integer, dimension(:), allocatable  :: atom_num ! Atomic numbers
    real(rp), dimension(:,:,:), allocatable :: atom_pos ! Atom positions
    real(rp), dimension(:,:,:,:,:), allocatable :: atom_lcl     ! Local axis
    logical, dimension(:), allocatable  :: grid_imp ! Important grid positions

    ! Arrays containing just important grid positions and valeues
    real(rp), dimension(:,:), allocatable   :: grid_pos ! Import. Grid positions
    real(rp), dimension(:), allocatable :: grid_val ! Import. Grid values
    real(rp), dimension(:), allocatable :: grid_fit ! Import. Grid values fit
    integer, dimension(:), allocatable  :: grid_aff ! Grid point affiliation
    integer, dimension(:), allocatable  :: grid_sid ! Grid point start index

    ! Global variables and arrays contain full information of one cube file
    integer :: file_nmx, file_ngx, file_ngy, file_ngz
    real(rp), dimension(3)  :: file_org
    real(rp), dimension(3)  :: file_axx, file_axy, file_axz
    real(rp), dimension(:), allocatable :: file_val
    real(rp), dimension(:), allocatable :: file_fit
    real(rp), dimension(:,:), allocatable   :: file_grd

    ! MDCM information
    integer :: Nchgs    ! Number of total MDCM charges to fit
    integer :: Nfrms    ! Number of local axis frames
    integer :: Nqdim    ! Length of MDCM charges array
    integer :: Npdim    ! Length of polynomial charges array
    integer :: Npoly    ! Number of polynomial coefficients
    real(rp)    :: Qtotl    ! Total sum of MDCM charges
    character(len=1024) :: mdcm_rlab    ! MDCM residue label
    real(rp), dimension(:), allocatable :: mdcm_clcl    ! Local MDCM charges
    real(rp), dimension(:), allocatable :: mdcm_poly    ! Poly. coefficients
    integer, dimension(:,:), allocatable    :: mdcm_afrm    ! Local frame atom
    integer, dimension(:), allocatable  :: mdcm_ftyp    ! Z-axis type of frame
    integer, dimension(:,:), allocatable    :: mdcm_nchg    ! Number of charges
                                                            ! per atom in frame
    real(rp), dimension(:), allocatable :: mdcm_cxyz    ! Global MDCM charges
    logical :: use_poly = .false.

    ! Fitting information
    integer :: Nafit    ! Number of system atoms to fit charges on
    integer, dimension(:), allocatable  :: atom_fit ! Respective atom indices

    ! Grid point selection parameter
    ! Radius smaller than this vdw_radius gets ignored
    real(rp) :: vdw_grid_min_cutoff = 1.20_rp
    ! Radius larger than this vdw_radius gets ignored
    real(rp) :: vdw_grid_max_cutoff = 2.20_rp


    contains

    !///////////////////////////////////////////////////////////////////////////
    !
    !   Initialize reference ESP and density data from a list of
    !   multiple cube files
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine load_cube_files( &
        Nfiles,                 &
        Nchar,                  &
        files_esp_cube,         &
        files_dens_cube)

        implicit none

        ! Input
        !-------
        integer :: Nfiles, Nchar
        character(len=Nchar), dimension(:) ::   files_esp_cube(Nfiles), &
                                                files_dens_cube(Nfiles)
!f2py   intent(in) Nfiles, Nchar
!f2py   intent(in) files_esp_cube, files_dens_cube
!f2py   depend(Nfiles) files_esp_cube, files_dens_cube

        ! Variables
        !-----------

        integer :: i, j, Igrid, Iimp

        ! System information
        integer :: fNatom, fNgrid, fNimpt
        integer, dimension(:), allocatable  :: fatom_num
        real(rp), dimension(:,:), allocatable    :: fatom_pos

        ! Read atom number
        call read_natm_file(    &
            files_esp_cube(1),  &
            fNatom)

        Natom = fNatom

        ! Allocate and initialize global system information arrays
        if(.not. allocated(fatom_num)) allocate(fatom_num(Natom))
        if(.not. allocated(fatom_pos)) allocate(fatom_pos(3, Natom))
        if(.not. allocated(atom_num)) allocate(atom_num(Natom))
        if(.not. allocated(atom_pos)) allocate(atom_pos(3, Natom, Nfiles))

        ! Get system information
        do i = 1, Nfiles

            ! Read ESP file
            call read_cube_file(    &
                files_esp_cube(i),  &
                1,                  &
                Igrid,              &
                Iimp,               &
                fNatom,             &
                fatom_num,          &
                fatom_pos,          &
                fNgrid,             &
                fNimpt)

            ! Check information
            if(Natom /= fNatom) call raise_error("Atom system deviation"//&
                " in file '"//trim(files_esp_cube(i))//"'.")

            if(i == 1) then
                atom_num = fatom_num
            else
                do j = 1, fNatom
                    if(atom_num(j) /= fatom_num(j)) call raise_error( &
                        "Atomic number deviation in file '"//&
                        trim(files_esp_cube(i))//"'.")
                end do
            end if

            ! Add atom positions
            atom_pos(:, :, i) = fatom_pos(:, :)

            ! Add grid counter
            if(i == 1) then
                Ngrid = fNgrid
                file_nmx = fNgrid
            else
                Ngrid = Ngrid + fNgrid
                if(file_nmx <= fNgrid) file_nmx = fNgrid
            end if

        end do

        ! Fit parameters
        Nafit = Natom
        if(.not. allocated(atom_fit)) allocate(atom_fit(Natom))
        do i = 1, Natom
            atom_fit(i) = i
        end do

        ! Allocate and initialize global system information arrays
        if(.not. allocated(grid_imp)) allocate(grid_imp(Ngrid))
        if(.not. allocated(file_grd)) allocate(file_grd(3, file_nmx))
        if(.not. allocated(file_val)) allocate(file_val(file_nmx))
        if(.not. allocated(file_fit)) allocate(file_fit(file_nmx))

        Nimpt = 0

        ! Determine grid point importance
        Igrid = 0
        do i = 1, Nfiles

            ! Read ESP file
            call read_cube_file(    &
                files_esp_cube(i),  &
                2,                  &
                Igrid,              &
                Iimp,               &
                fNatom,             &
                fatom_num,          &
                fatom_pos,          &
                fNgrid,             &
                fNimpt)

            ! Increment grid counter
            Igrid = Igrid + fNgrid

            ! Increment important grid counter
            Nimpt = Nimpt + fNimpt

        end do

        ! Allocate and initialize global system information arrays
        if(.not. allocated(grid_pos)) allocate(grid_pos(3, Nimpt))
        if(.not. allocated(grid_val)) allocate(grid_val(Nimpt))
        if(.not. allocated(grid_fit)) allocate(grid_fit(Nimpt))
        if(.not. allocated(grid_aff)) allocate(grid_aff(Nimpt))
        if(.not. allocated(grid_sid)) allocate(grid_sid(Nfiles + 1))

        ! Determine important grid point
        Igrid = 0
        Iimp = 0
        grid_sid(1) = 1
        do i = 1, Nfiles

            ! Read ESP file
            call read_cube_file(    &
                files_esp_cube(i),  &
                3,                  &
                Igrid,              &
                Iimp,               &
                fNatom,             &
                fatom_num,          &
                fatom_pos,          &
                fNgrid,             &
                fNimpt)

            ! Grid point affiliation
            do j = Iimp + 1, Iimp + fNimpt
                grid_aff(j) = i
            end do

            ! Increment grid counter
            Igrid = Igrid + fNgrid
            Iimp = Iimp + fNimpt

            ! Grid point start index
            grid_sid(i + 1) = Iimp + 1

        end do

        ! Store list of loaded ESP files
        Nload = Nfiles
        if(.not. allocated(files_load_cube)) allocate(files_load_cube(Nfiles))
        files_load_cube = files_esp_cube

    end subroutine load_cube_files


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Just read atom number from cube file
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine read_natm_file(  &
        file_cube,              &
        fNatom)

        implicit none

        ! Input
        !-------

        character(len=*), intent(in)    :: file_cube

        ! Output
        !--------

        integer, intent(out)    :: fNatom

        ! Variables
        !-----------

        integer :: ios
        real(rp), dimension(3)  :: forigin
        character(len=128)  :: ctmp


        if(verbose) write(*,'(A,I5)') "Reading '"//trim(file_cube)//"'..."//&
            " just atom number "

        ! Open cube files
        open(30, file=trim(file_cube), status="old", action="read", iostat=ios)
        if(ios/=0) call raise_error("Could not open '"//trim(file_cube)//"'.")

        ! Cube file header
        read(30,'(A128)', iostat=ios) ctmp
        if(ios/=0) call raise_error("Could not read '"//trim(file_cube)//"':"//&
            " Header")
        read(30,'(A128)', iostat=ios) ctmp
        if(ios/=0) call raise_error("Could not read '"//trim(file_cube)//"':"//&
            " Header")

        ! Read coordinate system information and atoms number
        read(30,*,iostat=ios) fNatom, forigin(:)
        if(ios/=0) call raise_error("Could not read '"//trim(file_cube)//"':"//&
            " Atom number or coordinate system information.")

        ! Close cube file
        close(30)

    end subroutine read_natm_file


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Read cube file
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine read_cube_file(  &
        file_cube,              &
        mode,                   &
        Igrid,                  &
        Iimp,                   &
        fNatom,                 &
        fatom_num,              &
        fatom_pos,              &
        fNgrid,                 &
        fNimpt)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        integer :: mode, Igrid, Iimp
        character(len=*), intent(in)    :: file_cube

        ! Output
        !--------

        integer, intent(out)    :: fNatom, fNgrid, fNimpt
        integer, dimension(:), intent(out) :: fatom_num(Natom)
        real(rp), dimension(:,:), intent(out)   :: fatom_pos(3, Natom)

        ! Variables
        !-----------

        integer :: ios, idxgrid, idxline, idximpt, i, j, k
        real(rp), dimension(3)  :: x

        ! Dummy variables
        real(rp)    :: rtmp
        character(len=128)  :: ctmp

        ! Coordinate system
        real(rp), dimension(3)  :: forigin, faxisX, faxisY, faxisZ
        ! Coordinate system grid points
        integer :: fNgridX, fNgridY, fNgridZ


        if(verbose) write(*,'(A,I5)') "Reading '"//trim(file_cube)//"'..."//&
            " mode ", mode

        ! Open cube files
        open(30, file=trim(file_cube), status="old", action="read", iostat=ios)
        if(ios/=0) call raise_error("Could not open '"//trim(file_cube)//"'.")

        ! Cube file header
        read(30,'(A128)', iostat=ios) ctmp
        if(ios/=0) call raise_error("Could not read '"//trim(file_cube)//"':"//&
            " Header")
        read(30,'(A128)', iostat=ios) ctmp
        if(ios/=0) call raise_error("Could not read '"//trim(file_cube)//"':"//&
            " Header")

        ! Read coordinate system information and atoms number
        read(30,*,iostat=ios) fNatom, forigin(:)
        if(ios/=0) call raise_error("Could not read '"//trim(file_cube)//"':"//&
            " Atom number or coordinate system information.")
        read(30,*,iostat=ios) fNgridX, faxisX(:)
        if(ios/=0) call raise_error("Could not read '"//trim(file_cube)//"':"//&
            " Coordinate system information in X.")
        read(30,*,iostat=ios) fNgridY, faxisY(:)
        if(ios/=0) call raise_error("Could not read '"//trim(file_cube)//"':"//&
            " Coordinate system information in Y.")
        read(30,*,iostat=ios) fNgridZ, faxisZ(:)
        if(ios/=0) call raise_error("Could not read '"//trim(file_cube)//"':"//&
            " Coordinate system information in Z.")

        ! Read atom information
        do i = 1, Natom
            read(30,*,iostat=ios) fatom_num(i), rtmp, fatom_pos(:,i)
            if(ios/=0) call raise_error("Could not read '"//trim(file_cube)//&
                "': Atom information.")
        end do

        ! Total number of grid points
        fNgrid = fNgridX*fNgridY*fNgridZ

        ! If requested, store grid information of cube file
        if(mode == 4) then

            ! Origin
            file_org = forigin(:)

            ! Grid points per axis
            file_ngx = fNgridX
            file_ngy = fNgridY
            file_ngz = fNgridZ

            ! Grid axis
            file_axx = faxisX(:)
            file_axy = faxisY(:)
            file_axz = faxisZ(:)

        end if

        ! This part is to execute at the second run
        if(mode > 1) then

            ! Read grid points and identify important ones
            idxgrid = 0
            idximpt = 0
            idxline = 0
            do i = 0, fNgridX - 1
                do j = 0, fNgridY - 1
                    do k = 0, fNgridZ - 1

                        ! Current grid point counter
                        idxgrid = idxgrid + 1

                        ! Increment line counter
                        idxline = idxline + 1

                        ! Current grid position
                        x = forigin + i*faxisX + j*faxisY + k*faxisZ

                        ! This part is to execute at the second run
                        ! Detect important grid points in file
                        if(mode == 2) then

                            ! Check grid point importance
                            if(in_interaction_belt(     &
                                fatom_pos,              &
                                x,                      &
                                vdw_grid_min_cutoff,    &
                                vdw_grid_max_cutoff)) then

                                ! Increment important grid point counter
                                idximpt = idximpt + 1

                                ! Label grid point as important
                                grid_imp(Igrid + idxgrid) = .true.

                            else

                                ! Label grid point as not important
                                grid_imp(Igrid + idxgrid) = .false.

                            end if

                        end if

                        ! This part is to execute at the third run
                        ! Read important grid points in file
                        if(mode == 3) then

                            if(grid_imp(Igrid + idxgrid)) then

                                ! Increment important grid point counter
                                idximpt = idximpt + 1

                                ! Store important grid point
                                grid_pos(:, Iimp + idximpt) = x

                                ! Read respective grid value
                                read(30,'(ES13.5)',advance='no',iostat = ios) &
                                    grid_val(Iimp + idximpt)
                                if(ios/=0) call raise_error("Could not read '"&
                                    //trim(file_cube)//"': Grid information.")

                            else

                                ! Read respective grid value
                                read(30,'(ES13.5)',advance='no',iostat = ios) &
                                    rtmp
                                if(ios/=0) call raise_error("Could not read '"&
                                    //trim(file_cube)//"': Grid information.")

                            end if

                        end if

                        ! This part is to execute if all grid points and grid
                        ! values are requested and stored in global array
                        ! file_grd and file_val of length fNgrid
                        ! Read all grid points in file
                        if(mode == 4) then

                            ! Store grid point
                            file_grd(:, idxgrid) = x

                            ! Read respective grid value
                            read(30,'(ES13.5)',advance='no',iostat = ios) &
                                file_val(idxgrid)
                            if(ios/=0) call raise_error("Could not read '"&
                                //trim(file_cube)//"': Grid information.")

                        end if

                        ! Sometimes lines have to be skipped
                        if(mod(idxline, 6) == 0 .or. &
                            mod(idxline, fNgridZ) == 0) then
                            read(30,*)  ! line break
                            if(mod(idxline, fNgridZ) == 0) idxline = 0
                        end if

                    end do
                end do
            end do

            ! Number of important grid points
            fNimpt = idximpt

        end if

        ! Close cube file
        close(30)

    end subroutine read_cube_file


    !///////////////////////////////////////////////////////////////////////////
    !
    ! Checks whether a point is inside the interaction belt as defined by the
    ! vdW radii
    !
    !///////////////////////////////////////////////////////////////////////////
    logical function in_interaction_belt(fatom_pos, x, mincut, maxcut)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        real(rp), dimension(:), intent(in)  :: x(3)
        real(rp), dimension(:), intent(in)  :: fatom_pos(3, Natom)
        ! Minimum and maximum grid cut-offs
        real(rp), intent(in)    :: mincut, maxcut

        ! Variables
        !-----------

        integer :: i, a, b
        real(rp)    :: r, rmin

        ! First check we're not inside the molecule
        rmin = 420._rp
        do a = 1, Natom

            r = sqrt(sum((fatom_pos(:, a) - x)**2))/ &
                (vdw_radius(atom_num(a)))

            if(r <= rmin .or. a == 1) then
                rmin = r
            end if

        end do

        ! This means the radius is not in the defined interaction cutoff
        if(rmin <= mincut) then
            in_interaction_belt = .false.
            return
        end if

        ! Now check we're not too far from any relevant atoms
        rmin = 420._rp
        do b = 1, Nafit

            a = atom_fit(b)

            r = sqrt(sum((fatom_pos(:, a) - x)**2))/ &
                (vdW_radius(atom_num(a)))

            if(r <= rmin .or. b == 1) then
                rmin = r
            end if

        end do

        ! This means the radius is not in the defined interaction cutoff.
        ! If maxcut < 0, there is no maximum interaction cutoff
        if(maxcut > 0._rp) then
            if(rmin >= maxcut) then
                in_interaction_belt = .false.
                return
            end if
        end if

        ! If the grid point passed so far it is important
        in_interaction_belt = .true.

        return

    end function in_interaction_belt


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Load model charges from MDCM xyz file
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine load_cxyz_file(  &
        file_cxyz)

        implicit none

        ! Input
        !-------

        character(len=*)    :: file_cxyz
!f2py   intent(in) file_cxyz

        ! Variables
        !-----------

        integer :: ios, i

        ! Dummy variables
        real(rp)    :: rtmp
        character(len=128)  :: ctmp

        if(verbose) write(*,'(A)') "Reading '"//trim(file_cxyz)//"'..."

        ! Open MDCM charge file
        open(30, file=trim(file_cxyz), status="old", action="read", iostat=ios)

        ! Read number of MDCM charges
        read(30,*,iostat=ios) Nchgs
        if((ios/=0) .or. (Nchgs < 1)) call raise_error(   &
            "Could not read '"//trim(file_cxyz)//"': Wrong number of "//&
            "MDCM charges.")

        ! Check if global MDCM charges already allocated
        if(allocated(mdcm_cxyz)) then
            if(verbose) write(*,'(A)') "Global MDCM charges already allocated."
            if(.not. Nqdim == 4*Nchgs) call raise_error(   &
                "Wrong number of MDCM charges.")
        end if

        ! Allocate MDCM charge array
        Nqdim = 4*Nchgs
        if(.not. allocated(mdcm_cxyz)) allocate(mdcm_cxyz(Nqdim))

        ! Skip comment line
        read(30,*,iostat=ios)

        ! Read charge coordinates and magnitude
        Qtotl  = 0._rp
        do i = 1, Nqdim, 4

            ! Read local MDCM coordinates in local axis frame
            read(30,*) ctmp, mdcm_cxyz(i:i+3)
            Qtotl = Qtotl + mdcm_cxyz(i+3)

            ! Convert to positions in Bohr
            mdcm_cxyz(i:i+2) = mdcm_cxyz(i:i+2)*ang2a0

        end do

        ! Close MDCM file
        close(30)

        ! Store load model xyz file
        file_load_cxyz = file_cxyz

    end subroutine load_cxyz_file


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Load model charges from MDCM dcm file
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine load_clcl_file(  &
        file_clcl)

        implicit none

        ! Input
        !-------

        character(len=*)    :: file_clcl
!f2py   intent(in) file_clcl

        ! Variables
        !-----------

        integer :: ios, idxchrg, i, j, k
        character(len=2)    :: ftyp

        ! Dummy variables
        integer :: itmp
        real(rp)    :: rtmp
        real(rp), dimension(:)  :: rtmp4(4)
        character(len=128)  :: ctmp

        if(verbose) write(*,'(A)') "Reading '"//trim(file_clcl)//"'..."

        ! Open MDCM charge file
        open(30, file=trim(file_clcl), status="old", action="read", iostat=ios)

        ! Number of residue types in MDCM file - only first one is loaded
        read(30,*,iostat=ios) itmp
        if(ios/=0) call raise_error("Could not read '"//trim(file_clcl)//"':"//&
            " Missing number of residues.")

!         ! Comment line
!         read(30,*,iostat=ios) ctmp
!         write(*,*) ctmp
!         if(ios/=0) call raise_error("Could not read '"//trim(file_clcl)//"':"//&
!             " Wrong header.")

        ! Residue label
        read(30,*,iostat=ios) mdcm_rlab
        if(ios/=0) call raise_error("Could not read '"//trim(file_clcl)//"':"//&
            " Missing residue label.")

        ! Read number of local axis frames
        read(30,*,iostat=ios) Nfrms
        if(ios/=0) call raise_error("Could not read '"//trim(file_clcl)//"':"//&
            " Missing number of local axis frames.")

        ! Allocate MDCM arrays for the first iteration
        Nchgs = 0
        Qtotl = 0._rp
        if(.not. allocated(mdcm_afrm)) allocate(mdcm_afrm(3, Nfrms))
        if(.not. allocated(mdcm_nchg)) allocate(mdcm_nchg(3, Nfrms))
        if(.not. allocated(mdcm_ftyp)) allocate(mdcm_ftyp(Nfrms))

        ! Read frames
        do i = 1, Nfrms

            ! Read atom indices
            read(30,*,iostat=ios) mdcm_afrm(:, i), ftyp
            if(ios/=0) call raise_error("Could not read '"//trim(file_clcl)//&
                "': Missing frame atoms or type.")

            ! Convert frame type for z-axis (BI: 1, BO: 0)
            ! BI: Bisector z-axis, BO: atom 1-2 bond as z-axis
            if(ftyp=='BI') then
                mdcm_ftyp(i) = 1
                if(mdcm_afrm(3, i) == 0) call raise_error("Could not read '"//&
                    trim(file_clcl)//"': Bisector z-axis not avaiable for"//&
                    " diatomic frame. Use BO instead.")
            else if(ftyp=='BO') then
                mdcm_ftyp(i) = 0
            else
                if(ios/=0) call raise_error("Could not read '"//trim(file_clcl)&
                    //"': Wrong frame type.")
            end if

            ! Read frame atom information
            do j = 1, 3

                ! Read number of charges per atom in frame (and polarization)
                read(30,*,iostat=ios) mdcm_nchg(j, i), itmp
                if(ios/=0) call raise_error("Could not read '"//trim(file_clcl)//&
                    "': Missing number of charges per atom in frame.")
                Nchgs = Nchgs + mdcm_nchg(j, i)

                ! Skip MDCM charges
                do k = 1, mdcm_nchg(j, i)
                    read(30,*,iostat=ios) rtmp4
                end do

            end do

        end do

        ! Allocate MDCM arrays for the second iteration
        Nqdim = 4*Nchgs
        if(.not. allocated(mdcm_clcl)) allocate(mdcm_clcl(Nqdim))

        ! Rewind file
        rewind(30)
        do i = 1, 4
            read(30,*)  ! Header
        end do

        ! Read MDCM charges
        idxchrg = 1
        do i = 1, Nfrms

            read(30,*)  ! Frame information

            ! Read frame atom information
            do j = 1, 3

                read(30,*)  ! Charges

                ! Read MDCM charges for atom in frame
                do k = 1, mdcm_nchg(j, i)

                    read(30,*,iostat=ios) mdcm_clcl(idxchrg:idxchrg+3)
                    if(ios/=0) call raise_error("Could not read '"//&
                        trim(file_clcl)//"': Missing MDCM charges.")

                    ! Convert to positions in Bohr
                    mdcm_clcl(idxchrg:idxchrg+2) = &
                        mdcm_clcl(idxchrg:idxchrg+2)*ang2a0

                    ! Add charge to total charge counter
                    Qtotl = Qtotl + mdcm_clcl(idxchrg+3)

                    ! Increment index
                    idxchrg = idxchrg + 4

                end do

            end do

        end do

        ! Close MDCM file
        close(30)

        ! Store load MDCM file
        file_load_clcl = file_clcl

    end subroutine load_clcl_file


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Load model charges polynomial coefficients from MDCM dcm file
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine load_poly_file(  &
        file_poly)

        implicit none

        ! Input
        !-------

        character(len=*)    :: file_poly
!f2py   intent(in) file_poly

        ! Variables
        !-----------

        integer :: ios, idxchrg, i, j, k, l
        character(len=2)    :: ftyp

        ! Dummy variables
        integer :: itmp
        real(rp)    :: rtmp
        character(len=128)  :: ctmp

        if(verbose) write(*,'(A)') "Reading '"//trim(file_poly)//"'..."

        ! Open MDCM charge file
        open(30, file=trim(file_poly), status="old", action="read", iostat=ios)

        ! Number of residue types in MDCM file - only first one is loaded
        read(30,*,iostat=ios) itmp
        if(ios/=0) call raise_error("Could not read '"//trim(file_poly)//"':"//&
            " Missing number of residues.")

        ! Number of polynomial coefficients
        read(30,*,iostat=ios) Npoly
        if(ios/=0) call raise_error("Could not read '"//trim(file_poly)//"':"//&
            " Missing number of polynomial coefficients.")

        ! Comment line
!         read(30,*,iostat=ios) ctmp
!         if(ios/=0) call raise_error("Could not read '"//trim(file_poly)//"':"//&
!             " Wrong header.")

        ! Residue label
        read(30,*,iostat=ios) mdcm_rlab
        if(ios/=0) call raise_error("Could not read '"//trim(file_poly)//"':"//&
            " Missing residue label.")

        ! Read number of local axis frames
        read(30,*,iostat=ios) Nfrms
        if(ios/=0) call raise_error("Could not read '"//trim(file_poly)//"':"//&
            " Missing number of local axis frames.")

        ! Allocate MDCM arrays for the first iteration
        Nchgs = 0
        Qtotl = 0._rp
        if(.not. allocated(mdcm_afrm)) allocate(mdcm_afrm(3, Nfrms))
        if(.not. allocated(mdcm_nchg)) allocate(mdcm_nchg(3, Nfrms))
        if(.not. allocated(mdcm_ftyp)) allocate(mdcm_ftyp(Nfrms))

        ! Read frames
        do i = 1, Nfrms

            ! Read atom indices
            read(30,*,iostat=ios) mdcm_afrm(:, i), ftyp
            if(ios/=0) call raise_error("Could not read '"//trim(file_poly)//&
                "': Missing frame atoms or type.")

            ! Convert frame type for z-axis (BI: 1, BO: 0)
            ! BI: Bisector z-axis, BO: atom 1-2 bond as z-axis
            if(ftyp=='BI') then
                mdcm_ftyp(i) = 1
                if(mdcm_afrm(3, i) == 0) call raise_error("Could not read '"//&
                    trim(file_poly)//"': Bisector z-axis not avaiable for"//&
                    " diatomic frame. Use BO instead.")
            else if(ftyp=='BO') then
                mdcm_ftyp(i) = 0
            else
                if(ios/=0) call raise_error("Could not read '"//trim(file_poly)&
                    //"': Wrong frame type.")
            end if

            ! Read frame atom information
            do j = 1, 3

                ! Read number of charges per atom in frame (and polarization)
                read(30,*,iostat=ios) mdcm_nchg(j, i), itmp
                if(ios/=0) call raise_error("Could not read '"//&
                    trim(file_poly)//&
                    "': Missing number of charges per atom in frame.")
                Nchgs = Nchgs + mdcm_nchg(j, i)

                ! Skip MDCM charges
                do k = 1, mdcm_nchg(j, i)*4
                    read(30,*)
                end do

            end do

        end do

        ! Allocate MDCM arrays for the second iteration
        Nqdim = 4*Nchgs
        Npdim = (Npoly*3 + 1)*Nchgs
        if(.not. allocated(mdcm_poly)) allocate(mdcm_poly(Npdim))

        ! Rewind file
        rewind(30)
        do i = 1, 5
            read(30,*)  ! Header
        end do

        ! Read MDCM charges
        idxchrg = 1
        do i = 1, Nfrms

            read(30,*)  ! Frame information

            ! Read frame atom information
            do j = 1, 3

                read(30,*)  ! Charges

                ! Read MDCM charges for atom in frame
                do k = 1, mdcm_nchg(j, i)

                    ! Read polynomial coefficients for each axis
                    do l = 1, 3

                        ! Read coefficients
                        read(30,*,iostat=ios) mdcm_poly(idxchrg:idxchrg+Npoly-1)
                        if(ios/=0) call raise_error("Could not read '"//&
                            trim(file_poly)//"': Missing MDCM coefficients.")

                        ! Convert to positions in Bohr
                        mdcm_poly(idxchrg:idxchrg+Npoly-1) = &
                            mdcm_poly(idxchrg:idxchrg+Npoly-1)*ang2a0

                        ! Increment index
                        idxchrg = idxchrg + Npoly

                    end do

                    ! Read charge
                    read(30,*,iostat=ios) mdcm_poly(idxchrg)
                    if(ios/=0) call raise_error("Could not read '"//&
                        trim(file_poly)//"': Missing MDCM charges.")

                    ! Add charge to total charge counter
                    Qtotl = Qtotl + mdcm_poly(idxchrg+1)

                    ! Increment index
                    idxchrg = idxchrg + 1

                end do

            end do

        end do

        ! Close MDCM file
        close(30)

        ! Store load MDCM file
        file_load_poly = file_poly

        ! Set polynomial flag
        use_poly = .true.

    end subroutine load_poly_file


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Compute local axis frame for loaded structures
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine calc_axis_locl_all()

        implicit none

        ! Variables
        !-----------

        integer :: i

        if(verbose) write(*,'(A)') "Start computing local axis systems ..."

        ! Allocate local axis system
        if(.not. allocated(atom_lcl)) allocate(atom_lcl(3, 3, 3, Nfrms, Nload))

        ! Iterate over loaded files
        do i = 1, Nload

            ! Compute local axis system
            call calc_axis_locl_pos(    &
                atom_pos(:, :, i),      &
                atom_lcl(:, :, :, :, i))

        end do

    end subroutine calc_axis_locl_all


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Compute ESP of MDCM model at grid point and with given atom positions
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine calc_axis_locl_pos(fatom_pos, fatom_elcl)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        real(rp), dimension(:), intent(in)  :: fatom_pos(3, Natom)

        ! Output
        !--------

        real(rp), dimension(:,:,:,:), intent(out) :: fatom_elcl(3, 3, 3, Nfrms)

        ! Variables
        !-----------

        integer :: i, a1, a2, a3
        real(rp)    :: l21, l23, l123
        real(rp), dimension(:)  :: r21(3), r23(3), n123(3)

        ! Iterate over frames
        do i = 1, Nfrms

            ! Frame atoms
            a1 = mdcm_afrm(1, i)
            a2 = mdcm_afrm(2, i)
            a3 = mdcm_afrm(3, i)

            ! Set local axis zero
            fatom_elcl(:, :, :, i) = 0._rp

            ! That is all for monoatomic case
            if(a2 == 0 .and. a3 == 0) then
                cycle
            end if

            ! Atom 2->1 bond
            r21 = fatom_pos(:, a1) - fatom_pos(:, a2)
            l21 = sqrt(r21(1)**2 + r21(2)**2 + r21(3)**2)

            ! Atom 2->3 bond
            r23 = fatom_pos(:, a3) - fatom_pos(:, a2)
            l23 = sqrt(r23(1)**2 + r23(2)**2 + r23(3)**2)

            ! If atom 1-2 bond z-axis
            if(mdcm_ftyp(i) == 0) then

                ! Local z-axis atom 1 and 2: r(a2->a1)
                fatom_elcl(:, 3, 1, i) = r21/l21
                fatom_elcl(:, 3, 2, i) = fatom_elcl(:, 3, 1, i)

                ! That is all for diatomic case
                if(a3 == 0) then
                    cycle
                end if

                ! Local z-axis atom 3: r(a2->a3)
                fatom_elcl(:, 3, 3, i) = r23/l23

                ! Normal vector r(a2->a1) x r(a2->a3)
                n123(1) = r21(2)*r23(3) - r21(3)*r23(2)
                n123(2) = r21(3)*r23(1) - r21(1)*r23(3)
                n123(3) = r21(1)*r23(2) - r21(2)*r23(1)
                l123 = sqrt(n123(1)**2 + n123(2)**2 + n123(3)**2)

                ! In case of colinear arrangement
                if(l123 <= eps) then
                    cycle
                end if

                ! Local y-axis atom 1, 2 and 3: n(a1<-a2->a3)
                fatom_elcl(:, 2, 1, i) = n123/l123
                fatom_elcl(:, 2, 2, i) = fatom_elcl(:, 2, 1, i)
                fatom_elcl(:, 2, 3, i) = fatom_elcl(:, 2, 1, i)

                ! Local x-axis atom 1, 2 equal
                ! Cross product local z-axis atom 1 x local y-axis atom 1
                fatom_elcl(1, 1, 1, i) = &  ! a1exx = a1eyz*a1ezy - a1eyy*a1ezz
                    fatom_elcl(3, 2, 1, i)*fatom_elcl(2, 3, 1, i) &
                    - fatom_elcl(2, 2, 1, i)*fatom_elcl(3, 3, 1, i)
                fatom_elcl(2, 1, 1, i) = &  ! a1exy = a1eyx*a1ezz - a1eyz*a1ezx
                    fatom_elcl(1, 2, 1, i)*fatom_elcl(3, 3, 1, i) &
                    - fatom_elcl(3, 2, 1, i)*fatom_elcl(1, 3, 1, i)
                fatom_elcl(3, 1, 1, i) = &  ! a1exz = a1eyy*a1ezx - a1eyx*a1ezy
                    fatom_elcl(2, 2, 1, i)*fatom_elcl(1, 3, 1, i) &
                    - fatom_elcl(1, 2, 1, i)*fatom_elcl(2, 3, 1, i)

                fatom_elcl(:, 1, 2, i) = fatom_elcl(:, 1, 1, i)   ! a2ex = a1ex

                ! Local x-axis atom 3 equal
                ! Cross product local z-axis atom 3 x local y-axis atom 3
                fatom_elcl(1, 1, 3, i) = &  ! a3exx = a3eyz*a3ezy - a3eyy*a3ezz
                    fatom_elcl(3, 2, 3, i)*fatom_elcl(2, 3, 3, i) &
                    - fatom_elcl(2, 2, 3, i)*fatom_elcl(3, 3, 3, i)
                fatom_elcl(2, 1, 3, i) = &  ! a3exy = a3eyx*a3ezz - a3eyz*a3ezx
                    fatom_elcl(1, 2, 3, i)*fatom_elcl(3, 3, 3, i) &
                    - fatom_elcl(3, 2, 3, i)*fatom_elcl(1, 3, 3, i)
                fatom_elcl(3, 1, 3, i) = &  ! a3exz = a3eyy*a3ezx - a3eyx*a3ezy
                    fatom_elcl(2, 2, 3, i)*fatom_elcl(1, 3, 3, i) &
                    - fatom_elcl(1, 2, 3, i)*fatom_elcl(2, 3, 3, i)

            ! If bivector z-axis
            else

                ! Local z-axis atom 1 : r(a2->a1)
                fatom_elcl(:, 3, 1, i) = r21/l21

                ! Local z-axis atom 3 : r(a2->a3)
                fatom_elcl(:, 3, 3, i) = r23/l23

                ! Bisector of r(a2->a1) & r(a2->a3)
                n123 = r21 + r23
                l123 = sqrt(n123(1)**2 + n123(2)**2 + n123(3)**2)

                ! In case of colinear arrangement
                if(l123 <= eps) then
                    fatom_elcl(:, 3, 2, i) = fatom_elcl(:, 3, 1, i)
                    cycle
                end if

                ! Local z-axis atom 2 equal
                ! Bisector of r(a2->a1) & r(a2->a3)
                fatom_elcl(:, 3, 2, i) = n123/l123

                ! Normal vector r(a2->a1) x r(a2->a3)
                n123(1) = r21(2)*r23(3) - r21(3)*r23(2)
                n123(2) = r21(3)*r23(1) - r21(1)*r23(3)
                n123(3) = r21(1)*r23(2) - r21(2)*r23(1)
                l123 = sqrt(n123(1)**2 + n123(2)**2 + n123(3)**2)

                ! Local y-axis atom 1, 2 and 3: n(a1<-a2->a3)
                fatom_elcl(:, 2, 1, i) = n123/l123
                fatom_elcl(:, 2, 2, i) = fatom_elcl(:, 2, 1, i)
                fatom_elcl(:, 2, 3, i) = fatom_elcl(:, 2, 1, i)

                ! Local x-axis atom 1 equal
                ! Cross product local z-axis atom 1 x local y-axis atom 1
                fatom_elcl(1, 1, 1, i) = &  ! a1exx = a1eyz*a1ezy - a1eyy*a1ezz
                    fatom_elcl(3, 2, 1, i)*fatom_elcl(2, 3, 1, i) &
                    - fatom_elcl(2, 2, 1, i)*fatom_elcl(3, 3, 1, i)
                fatom_elcl(2, 1, 1, i) = &  ! a1exy = a1eyx*a1ezz - a1eyz*a1ezx
                    fatom_elcl(1, 2, 1, i)*fatom_elcl(3, 3, 1, i) &
                    - fatom_elcl(3, 2, 1, i)*fatom_elcl(1, 3, 1, i)
                fatom_elcl(3, 1, 1, i) = &  ! a1exz = a1eyy*a1ezx - a1eyx*a1ezy
                    fatom_elcl(2, 2, 1, i)*fatom_elcl(1, 3, 1, i) &
                    - fatom_elcl(1, 2, 1, i)*fatom_elcl(2, 3, 1, i)

                ! Local x-axis atom 2 equal
                ! Cross product local z-axis atom 2 x local y-axis atom 2
                fatom_elcl(1, 1, 2, i) = &  ! a2exx = a2eyz*a2ezy - a2eyy*a2ezz
                    fatom_elcl(3, 2, 2, i)*fatom_elcl(2, 3, 2, i) &
                    - fatom_elcl(2, 2, 2, i)*fatom_elcl(3, 3, 2, i)
                fatom_elcl(2, 1, 2, i) = &  ! a2exy = a2eyx*a2ezz - a2eyz*a2ezx
                    fatom_elcl(1, 2, 2, i)*fatom_elcl(3, 3, 2, i) &
                    - fatom_elcl(3, 2, 2, i)*fatom_elcl(1, 3, 2, i)
                fatom_elcl(3, 1, 2, i) = &  ! a2exz = a2eyy*a2ezx - a2eyx*a2ezy
                    fatom_elcl(2, 2, 2, i)*fatom_elcl(1, 3, 2, i) &
                    - fatom_elcl(1, 2, 2, i)*fatom_elcl(2, 3, 2, i)

                ! Local x-axis atom 3 equal
                ! Cross product local z-axis atom 3 x local y-axis atom 3
                fatom_elcl(1, 1, 3, i) = &  ! a3exx = a3eyz*a3ezy - a3eyy*a3ezz
                    fatom_elcl(3, 2, 3, i)*fatom_elcl(2, 3, 3, i) &
                    - fatom_elcl(2, 2, 3, i)*fatom_elcl(3, 3, 3, i)
                fatom_elcl(2, 1, 3, i) = &  ! a3exy = a3eyx*a3ezz - a3eyz*a3ezx
                    fatom_elcl(1, 2, 3, i)*fatom_elcl(3, 3, 3, i) &
                    - fatom_elcl(3, 2, 3, i)*fatom_elcl(1, 3, 3, i)
                fatom_elcl(3, 1, 3, i) = &  ! a3exz = a3eyy*a3ezx - a3eyx*a3ezy
                    fatom_elcl(2, 2, 3, i)*fatom_elcl(1, 3, 3, i) &
                    - fatom_elcl(1, 2, 3, i)*fatom_elcl(2, 3, 3, i)

            end if

        end do

    end subroutine calc_axis_locl_pos


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Convert MDCM charges from local to global
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine conv_clcl_cxyz(fatom_pos, fatom_elcl, fatom_cxyz)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        real(rp), dimension(:), intent(in)  :: fatom_pos(3, Natom)
        real(rp), dimension(:,:,:,:), intent(in) :: fatom_elcl(3, 3, 3, Nfrms)

        ! Output
        !--------

        real(rp), dimension(:), intent(out) :: fatom_cxyz(Nqdim)

        ! Variables
        !-----------

        integer :: idxchrg, i, j, k, l
        real(rp), dimension(:) :: x(3), fcta(3)

        ! MDCM charge counter
        idxchrg = 1

        fatom_cxyz = 0._rp

        ! Iterate over MDCM frames
        do i = 1, Nfrms

            ! Iterate over frame atoms
            do j = 1, 3

                ! Iterate of MDCM charges per frame atom
                do k = 1, mdcm_nchg(j, i)

                    ! Add atom position
                    fatom_cxyz(idxchrg:idxchrg + 2) =   &
                        fatom_pos(:, mdcm_afrm(j, i))

                    ! Iterate over local coordinates
                    do l = 1, 3

                        ! Convert charge from local to global axis system
                        x = mdcm_clcl(idxchrg + l - 1)      &
                            *fatom_elcl(:, l, j, i)
                        fatom_cxyz(idxchrg:idxchrg + 2) =   &
                            fatom_cxyz(idxchrg:idxchrg + 2) + x

                    end do

                    ! Set MDCM charge
                    fatom_cxyz(idxchrg + 3) = mdcm_clcl(idxchrg + 3)

                    ! Increment counter
                    idxchrg = idxchrg + 4

                end do

            end do

        end do

    end subroutine conv_clcl_cxyz


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Convert MDCM charges from polynomial to global
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine conv_poly_cxyz(fatom_pos, fatom_elcl, fatom_cxyz)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        real(rp), dimension(:), intent(in)  :: fatom_pos(3, Natom)
        real(rp), dimension(:,:,:,:), intent(in) :: fatom_elcl(3, 3, 3, Nfrms)

        ! Output
        !--------

        real(rp), dimension(:), intent(out) :: fatom_cxyz(Nqdim)

        ! Variables
        !-----------

        integer :: idxchrg, jdxpoly, i, j, k, l, m
        real(rp) :: cosa, xa,  fa
        real(rp), dimension(:) :: x(3)

        ! MDCM charge counter
        idxchrg = 1

        ! Polynomial coefficient counter
        jdxpoly = 1

        ! Global atom positions
        fatom_cxyz = 0._rp

        ! Iterate over MDCM frames
        do i = 1, Nfrms

            ! Compute local frame cosine of angle between
            ! local z-axis (3) of atom 1 and 3 of frame i
            cosa = sum(fatom_elcl(:, 3, 1, i)*fatom_elcl(:, 3, 3, i))

            ! Compute argument for polynomial
            xa = (1._rp - cosa**2)

            ! Iterate over frame atoms
            do j = 1, 3

                ! Iterate of MDCM charges per frame atom
                do k = 1, mdcm_nchg(j, i)

                    ! Add atom position
                    fatom_cxyz(idxchrg:idxchrg + 2) =   &
                        fatom_pos(:, mdcm_afrm(j, i))

                    ! Iterate over local coordinates
                    do l = 1, 3

                        ! Compute polynomial
                        fa = mdcm_poly(jdxpoly + Npoly - 1)
                        do m = 2, Npoly

                            fa = mdcm_poly(jdxpoly + Npoly - m) + fa*xa

                        end do

                        ! Increment counter
                        jdxpoly = jdxpoly + Npoly

!                         ! Check for max. charge offset
!                         if(fa .gt. maxlcl) then
!                             fa = maxlcl
!                         end if

                        ! Convert charge from local to global axis system
                        mdcm_clcl(idxchrg+l-1) = fa;
                        x = fa*fatom_elcl(:, l, j, i)
                        fatom_cxyz(idxchrg:idxchrg + 2) =   &
                            fatom_cxyz(idxchrg:idxchrg + 2) + x

                    end do

                    ! Set MDCM charge
                    fatom_cxyz(idxchrg + 3) = mdcm_poly(jdxpoly)
                    mdcm_clcl(idxchrg + 3) = mdcm_poly(jdxpoly)
                    ! Increment counter
                    idxchrg = idxchrg + 4
                    jdxpoly = jdxpoly + 1

                end do

            end do

        end do

    end subroutine conv_poly_cxyz


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Write MDCM charge files from local to global of loaded structures
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine write_cxyz_files()

        implicit none

        ! Variables
        !-----------

        integer :: i, diff
        real(rp)    :: rmse, mae, maxae
        real(rp), dimension(:)  :: fatom_cxyz(Nqdim)
        character(len=1024) ::  file_cxyz

        ! Compute local axis system if not done yet
        if(.not. allocated(atom_lcl)) call calc_axis_locl_all()

        if(.not. allocated(mdcm_clcl)) allocate(mdcm_clcl(Nqdim))

        ! Iterate over loaded files
        do i = 1, Nload

            ! Convert MDCM charges from local to global
            if(use_poly) then
                call conv_poly_cxyz(            &
                    atom_pos(:, :, i),          &
                    atom_lcl(:, :, :, :, i),    &
                    fatom_cxyz)
            else
                call conv_clcl_cxyz(            &
                    atom_pos(:, :, i),          &
                    atom_lcl(:, :, :, :, i),    &
                    fatom_cxyz)
            end if

            ! Calculate MDCM ESP on grid
            diff = grid_sid(i + 1) - grid_sid(i)
            call calc_mdcm_grid(                                &
                fatom_cxyz,                                     &
!                 mdcm_cxyz,                                      &
                diff,                                           &
                grid_pos(:, grid_sid(i):(grid_sid(i + 1) - 1)), &
                grid_fit(grid_sid(i):(grid_sid(i + 1) - 1)))

            ! Compute error between reference and MDCM ESP at important grid
            call calc_error(                                    &
                diff,                                           &
                grid_val(grid_sid(i):(grid_sid(i + 1) - 1)),    &
                grid_fit(grid_sid(i):(grid_sid(i + 1) - 1)),    &
                rmse,                                           &
                mae,                                            &
                maxae)

            ! Target file name
            file_cxyz = trim(files_load_cube(i))//".mdcm.xyz"

            ! Write MDCM charge in global coordinates to file
            call write_cxyz_file(   &
                file_cxyz,          &
                fatom_cxyz,         &
                !atom_lcl(:,:,:,:,i) &
                rmse,               &
                mae,                &
                maxae)

        end do

    end subroutine write_cxyz_files


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Write MDCM charge file
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine write_cxyz_file(file_cxyz, fatom_cxyz, rmse, mae, maxae)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        character(len=*), intent(in) :: file_cxyz
        real(rp), dimension(:), intent(in)  ::  fatom_cxyz(Nqdim)
        !real(rp), dimension(:), intent(in)  :: lcl(3, 3, 3, Nfrms)
        real(rp), intent(in)    :: rmse, mae, maxae

        ! Variables
        !-----------

        integer :: i, ios

        ! Open file
        open(30, file=trim(file_cxyz), status="replace", action="write", &
            iostat=ios)
        if(ios/=0) call raise_error("Could not open '"//trim(file_cxyz) &
            //"' to write MDCM charges in global axis system.")

        ! Number of MDCM charges
        write(30,'(I0)') Nchgs

        ! Disclaimer
        write(30,'(A,4A26)')    "s","x[A]","y[A]","z[A]","q[e]"

        ! MDCM charge coordinates in Angstrom with charge values
        do i = 1, Nqdim, 4

            if(fatom_cxyz(i + 3) > 0._rp) then
                write(30,'(A,1X,4(F25.16,1X))') &
                    "N", fatom_cxyz(i:i + 2)*a02ang, fatom_cxyz(i + 3)
            else
                write(30,'(A,1X,4(F25.16,1X))') &
                    "O", fatom_cxyz(i:i + 2)*a02ang, fatom_cxyz(i + 3)
            end if

        end do

        ! MDCM fit error
        write(30,*)
        write(30,'(A,ES23.9,A)') "        RMSE ", rmse*ha2kcal, " kcal/mol"
        write(30,'(A,ES23.9,A)') "         MAE ", mae*ha2kcal, " kcal/mol"
        write(30,'(A,ES23.9,A)') "     max. AE ", maxae*ha2kcal," kcal/mol"
        write(30,*)

        ! Disclaimer
        write(30,'(A)') "Coordinates in bohr"
        write(30,'(A,4A26)')  "s", "x[bohr]", "y[bohr]", "z[bohr]", "q[e]"

        ! MDCM charge coordinates in Bohr with charge values
 !       do i = 1, Nqdim, 4
 !
         !   if(fatom_cxyz(i + 3) > 0._rp) then
          !      write(30,'(A,1X,4(F25.16,1X))') &
           !         "+", atom_lcl(i:i + 2), fatom_cxyz(i + 3)
           ! else
           !     write(30,'(A,1X,4(F25.16,1X))') &
           !         "-", atom_lcl(i:i + 2), fatom_cxyz(i + 3)
           ! end if
           !
       ! end do

        ! Disclaimer
        write(30,'(A)') "local coordinates in angstrom"
        write(30,'(A,4A26)')  "s", "x[a]", "y[a]", "z[a", "q[e]"
        do i = 1, Nqdim, 4
        write(30,'(A,1X,4(F25.16,1X))') "x", mdcm_clcl(i:i+3)
         !   if(fatom_cxyz(i + 3) > 0._rp) then
          !      write(30,'(A,1X,4(F25.16,1X))') &
          !          "+", fatom_cxyz(i:i + 2), fatom_cxyz(i + 3)
          !  else
          !      write(30,'(A,1X,4(F25.16,1X))') &
          !          "-", fatom_cxyz(i:i + 2), fatom_cxyz(i + 3)
          !  end if
!
        end do


        ! Close file
        close(30)

    end subroutine write_cxyz_file


!     !///////////////////////////////////////////////////////////////////////////
!     !
!     !   Return MDCM charges in local axis system
!     !
!     !///////////////////////////////////////////////////////////////////////////
!     subroutine get_clcl(clcl)
!
!         implicit none
!
!         integer, parameter :: rp = selected_real_kind(15)   ! python float
!
!         ! Output
!         !--------
!
!         real(rp), dimension(Nqdim)  :: clcl
! !f2py   real(rp) optional, intent(out), dimension(n)    :: clcl
!
!         ! Check if local MDCM charges already allocated
!         if(.not. (allocated(mdcm_clcl) .or. use_poly)) call raise_error(   &
!             "No local MDCM charges are stored.")
!
!         clcl = mdcm_clcl(:)
!
!     end subroutine get_clcl


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Set MDCM charges in local axis system
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine set_clcl(clcl)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        real(rp), dimension(:)  :: clcl(Nqdim)
!f2py   intent(in) clcl
!f2py   depend(Nqdim) clcl

        ! Check if local MDCM charges already allocated
        if(.not. (allocated(mdcm_clcl) .or. use_poly)) call raise_error(   &
            "Local MDCM charges are not initiated.")

        mdcm_clcl = clcl(:)

    end subroutine set_clcl


!     !///////////////////////////////////////////////////////////////////////////
!     !
!     !   Return MDCM charges in global axis system
!     !
!     !///////////////////////////////////////////////////////////////////////////
!     subroutine get_cxyz(cxyz)
!
!         implicit none
!
!         integer, parameter :: rp = selected_real_kind(15)   ! python float
!
!         ! Output
!         !--------
!
!         real(rp), dimension(:)  :: cxyz(Nqdim)
! !f2py   intent(out) cxyz
! !f2py   depend(Nqdim) cxyz
!
!         ! Check if global MDCM charges already allocated
!         if(.not. allocated(mdcm_cxyz)) call raise_error(   &
!             "No global MDCM charges are stored.")
!
!         cxyz = mdcm_cxyz(:)
!
!     end subroutine get_cxyz


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Set MDCM charges in global axis system
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine set_cxyz(cxyz)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        real(rp), dimension(:)  :: cxyz(Nqdim)
!f2py   intent(in) cxyz
!f2py   depend(Nqdim) cxyz

        ! Check if global MDCM charges already allocated
        if(.not. allocated(mdcm_cxyz)) call raise_error(   &
            "Global MDCM charges are not initiated.")

        mdcm_cxyz = cxyz(:)

    end subroutine set_cxyz


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Set MDCM polynomial coefficients and charges
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine set_poly(poly)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        real(rp), dimension(:)  :: poly(Npdim)
!f2py   intent(in) poly
!f2py   depend(Npdim) poly

        ! Check if local MDCM charges already allocated
        if(.not. use_poly) call raise_error(   &
            "Local MDCM polynomial coefficients are not initiated.")

        mdcm_poly = poly(:)

    end subroutine set_poly


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Return total MDCM Fit error for important grid points
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine get_rmse(rmse)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Output
        !--------

        real(rp), intent(out)   :: rmse
!f2py   intent(out) rmse

        ! Variables
        !-----------

        integer :: i, diff, nall
        real(rp)    :: irmse
        real(rp), dimension(:)  :: fatom_cxyz(Nqdim)

        ! Check if global MDCM charges already allocated
        if(.not. (allocated(mdcm_clcl) .or. use_poly)) call raise_error(  &
            "No MDCM model is initiated.")

        ! Check if local axis system frame is allocated
        if(.not. allocated(atom_lcl)) call calc_axis_locl_all()

        ! Iterate over loaded files
        rmse = 0._rp
        nall = 0
        do i = 1, Nload

            ! Convert MDCM charges from local to global
            if(use_poly) then
                call conv_poly_cxyz(            &
                    atom_pos(:, :, i),          &
                    atom_lcl(:, :, :, :, i),    &
                    fatom_cxyz)
            else
                call conv_clcl_cxyz(            &
                    atom_pos(:, :, i),          &
                    atom_lcl(:, :, :, :, i),    &
                    fatom_cxyz)
            end if
            ! Calculate MDCM ESP on grid
            diff = grid_sid(i + 1) - grid_sid(i)
            call calc_mdcm_grid(                                &
                fatom_cxyz,                                     &
                diff,                                           &
                grid_pos(:, grid_sid(i):(grid_sid(i + 1) - 1)), &
                grid_fit(grid_sid(i):(grid_sid(i + 1) - 1)))

            ! Compute error between reference and MDCM ESP at important grid
            irmse = calc_mse(                                   &
                diff,                                           &
                grid_val(grid_sid(i):(grid_sid(i + 1) - 1)),    &
                grid_fit(grid_sid(i):(grid_sid(i + 1) - 1)))

            ! Add error weighted by grid points
            rmse = rmse + irmse*real(diff, rp)

            ! Add number of grid points
            nall = nall + diff

        end do

        rmse = sqrt(rmse/real(nall, rp))*ha2kcal

    end subroutine get_rmse


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Return total MDCM Fit error for important grid points
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine get_rmse_weighted(Nwgt, wght, rmse)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        integer :: Nwgt
        real(rp), dimension(:)  :: wght(Nwgt)
!f2py   intent(in) Nwgt
!f2py   intent(in) wght
!f2py   depend(Nwgt) wght

        ! Output
        !--------

        real(rp), intent(out)   :: rmse
!f2py   intent(out) rmse

        ! Variables
        !-----------

        integer :: i, diff, nall
        real(rp)    :: irmse
        real(rp), dimension(:)  :: fatom_cxyz(Nqdim)

        ! Check if global MDCM charges already allocated
        if(.not. (allocated(mdcm_clcl) .or. use_poly)) call raise_error(   &
            "No MDCM model is initiated.")

        ! Check if local axis system frame is allocated
        if(.not. allocated(atom_lcl)) call calc_axis_locl_all()

        ! Iterate over loaded files
        rmse = 0._rp
        nall = 0
        do i = 1, Nwgt

            ! Convert MDCM charges from local to global
            if(use_poly) then
                call conv_poly_cxyz(            &
                    atom_pos(:, :, i),          &
                    atom_lcl(:, :, :, :, i),    &
                    fatom_cxyz)
            else
                call conv_clcl_cxyz(            &
                    atom_pos(:, :, i),          &
                    atom_lcl(:, :, :, :, i),    &
                    fatom_cxyz)
            end if

            ! Calculate MDCM ESP on grid
            diff = grid_sid(i + 1) - grid_sid(i)
            call calc_mdcm_grid(                                &
                fatom_cxyz,                                     &
                diff,                                           &
                grid_pos(:, grid_sid(i):(grid_sid(i + 1) - 1)), &
                grid_fit(grid_sid(i):(grid_sid(i + 1) - 1)))

            ! Compute error between reference and MDCM ESP at important grid
            irmse = calc_mse(                                   &
                diff,                                           &
                grid_val(grid_sid(i):(grid_sid(i + 1) - 1)),    &
                grid_fit(grid_sid(i):(grid_sid(i + 1) - 1)))

            ! Add error weighted by grid points
            rmse = rmse + irmse*real(diff, rp)/wght(i)

            ! Add number of grid points
            nall = nall + diff

        end do

        rmse = sqrt(rmse/real(nall, rp))*ha2kcal

    end subroutine get_rmse_weighted


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Return total MDCM Fit error for important grid points
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine get_rmse_each(Nwgt, rmse)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        integer :: Nwgt
!f2py   intent(in) Nwgt

        ! Output
        !--------

        real(rp), dimension(Nwgt), intent(out)   :: rmse
!f2py   intent(out) rmse
!f2py   depend(Nwgt) rmse

        ! Variables
        !-----------

        integer :: i, diff
        real(rp), dimension(:)  :: fatom_cxyz(Nqdim)

        ! Check if global MDCM charges already allocated
        if(.not. (allocated(mdcm_clcl) .or. use_poly)) call raise_error(   &
            "No MDCM model is initiated.")

        ! Check if local axis system frame is allocated
        if(.not. allocated(atom_lcl)) call calc_axis_locl_all()

        ! Iterate over loaded files
        rmse = 0._rp
        do i = 1, Nwgt

            ! Convert MDCM charges from local to global
            if(use_poly) then
                call conv_poly_cxyz(            &
                    atom_pos(:, :, i),          &
                    atom_lcl(:, :, :, :, i),    &
                    fatom_cxyz)
            else
                call conv_clcl_cxyz(            &
                    atom_pos(:, :, i),          &
                    atom_lcl(:, :, :, :, i),    &
                    fatom_cxyz)
            end if

            ! Calculate MDCM ESP on grid
            diff = grid_sid(i + 1) - grid_sid(i)
            call calc_mdcm_grid(                                &
                fatom_cxyz,                                     &
                diff,                                           &
                grid_pos(:, grid_sid(i):(grid_sid(i + 1) - 1)), &
                grid_fit(grid_sid(i):(grid_sid(i + 1) - 1)))

            ! Compute error between reference and MDCM ESP at important grid
            rmse(i) = calc_rmse(                                &
                diff,                                           &
                grid_val(grid_sid(i):(grid_sid(i + 1) - 1)),    &
                grid_fit(grid_sid(i):(grid_sid(i + 1) - 1)))

            ! Convert energy
            rmse(i) = rmse(i)*ha2kcal

        end do

    end subroutine get_rmse_each


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Write ESP cube files of MDCM model with regard to loaded structures
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine write_mdcm_cube_files()

        implicit none

        ! Variables
        !-----------

        integer :: i, j
        real(rp)    :: rmse, mae, maxae
        character(len=1024) ::  file_cube
        character(len=1024) ::  file_cmt1, file_cmt2

        ! System variables
        integer :: fNatom, fNgrid, fNimpt
        integer, dimension(:) :: fatom_num(Natom)
        real(rp), dimension(:)  :: fatom_cxyz(Nqdim)
        real(rp), dimension(:,:)    :: fatom_pos(3, Natom)

        ! Compute local axis system if not done yet
        if(.not. allocated(atom_lcl)) call calc_axis_locl_all()

        ! Iterate over loaded files
        do i = 1, Nload

            ! Target cube file name
            file_cube = trim(files_load_cube(i))//".mdcm.cube"

            ! Read system grid information to
            call read_cube_file(    &
                files_load_cube(i), &
                4,                  &   ! Mode 4: just read grid points
                0,                  &   ! not used in mode 4
                0,                  &   ! not used in mode 4
                fNatom,             &
                fatom_num,          &
                fatom_pos,          &
                fNgrid,             &
                fNimpt)

            ! Convert MDCM charges from local to global axis of system
            if(use_poly) then
                call conv_poly_cxyz(            &
                    atom_pos(:, :, i),          &
                    atom_lcl(:, :, :, :, i),    &
                    fatom_cxyz)
            else
                call conv_clcl_cxyz(            &
                    atom_pos(:, :, i),          &
                    atom_lcl(:, :, :, :, i),    &
                    fatom_cxyz)
            end if

            ! Calculate MDCM ESP on grid
            call calc_mdcm_grid(        &
                fatom_cxyz,             &
                fNgrid,                 &
                file_grd(:, :fNgrid),   &
                file_fit(:fNgrid))

            ! Compute error between reference and MDCM ESP at cube file grid
            ! At full grid except within molecule
            do j = 1, fNgrid
                if(.not. in_interaction_belt(   &
                    atom_pos(:, :, i),          &
                    file_grd(:, j),             &
                    vdw_grid_min_cutoff,        &
                    -1._rp)) then
                    file_val(j) = file_fit(j)
                end if
            end do

            call calc_error(        &
                fNgrid,             &
                file_val(:fNgrid),  &
                file_fit(:fNgrid),  &
                rmse,               &
                mae,                &
                maxae)

            ! Cube file comment line
            write(file_cmt1, '(A19,E18.9,A9)')  &
                "Fitted ESP - RMSE: ",          &
                rmse*ha2kcal,                   &
                " kcal/mol"
!             file_cmt1 = "Fitted ESP"
            write(file_cmt2, '(A22,I2,A8)') &
                " from MDCM model with ",   &
                Nchgs,                      &
                " charges"

            ! Write cube file of MDCM ESP fit
            call write_cube_file(       &
                file_cube,              &
                file_cmt1,              &
                file_cmt2,              &
                Natom,                  &
                fNgrid,                 &
                file_org,               &
                file_ngx,               &
                file_ngy,               &
                file_ngz,               &
                file_axx,               &
                file_axy,               &
                file_axz,               &
                atom_num,               &
                atom_pos(:, :, i),      &
                file_fit(:fNgrid))

        end do

    end subroutine write_mdcm_cube_files


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Write cube file of model ESP with respect to given cube file
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine write_cube_file( &
        file_cube,              &
        file_cmt1,              &
        file_cmt2,              &
        fNatom,                 &
        fNgrid,                 &
        forg,                   &
        fngx,                   &
        fngy,                   &
        fngz,                   &
        faxx,                   &
        faxy,                   &
        faxz,                   &
        fatom_num,              &
        fatom_pos,              &
        fvgrid)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        character(len=*), intent(in)    :: file_cube, file_cmt1, file_cmt2
        integer, intent(in) :: fNatom, fNgrid, fngx, fngy, fngz
        integer, dimension(:), intent(in)   :: fatom_num
        real(rp), dimension(:), intent(in)  :: forg(3)
        real(rp), dimension(:), intent(in)  :: faxx(3), faxy(3), faxz(3)
        real(rp), dimension(:), intent(in)  :: fvgrid(fNgrid)
        real(rp), dimension(:,:), intent(in)    :: fatom_pos(3, fNatom)

        ! Variables
        !-----------

        integer :: i, j, k, idxgrid, idxline, ios

        ! Open file
        open(30, file=trim(file_cube), status="replace", action="write", &
            iostat=ios)
        if(ios/=0) call raise_error("Could not open '"//trim(file_cube) &
            //"' to write cube file.")

        ! Write Header
        write(30,'(1X,A)') trim(file_cmt1)
        write(30,'(A)') trim(file_cmt2)

        ! Write grid information
        write(30,'(I5,3(F12.6),I5)') fNatom, forg, 1
        write(30,'(I5,3(F12.6))')    fngx, faxx
        write(30,'(I5,3(F12.6))')    fngy, faxy
        write(30,'(I5,3(F12.6))')    fngz, faxz

        ! Write system atom positions
        do i = 1, fNatom
            write(30,'(I5,4(F12.6))')   &
                fatom_num(i),           &
                real(fatom_num(i), rp), &
                fatom_pos(:, i)
        end do

        ! Write grid values
        idxgrid = 0
        idxline = 0
        do i = 1, fngx

            do j = 1, fngy
                do k = 1, fngz

                    ! Increment grid point and line counter
                    idxgrid = idxgrid + 1
                    idxline = idxline + 1

                    ! Write grid value
                    write(30,'(ES13.5)', advance='no') fvgrid(idxgrid)

                    ! Sometimes lines have to be skipped
                    if(mod(idxline, 6) == 0 .or. &
                        mod(idxline, fngz) == 0) then
                        write(30,*) ! line break
                        if(mod(idxline, fngz) == 0) idxline = 0
                    end if

                end do
            end do
        end do

        ! Close file
        close(30)

    end subroutine write_cube_file


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Compute ESP of MDCM model on grid
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine calc_mdcm_grid(fatom_cxyz, fNgrid, fxgrid, fvgrid)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        integer, intent(in) ::  fNgrid
        real(rp), dimension(:), intent(in)  :: fatom_cxyz(Nqdim)
        real(rp), dimension(:), intent(in)  :: fxgrid(3, fNgrid)

        ! Output
        !--------

        real(rp), dimension(:,:), intent(out)    :: fvgrid(fNgrid)

        ! Variables
        !-----------

        integer :: i
        real(rp), dimension(:)  :: r(3)

        ! Iterate over grid points
        fvgrid = 0._rp
        do i = 1, fNgrid

            ! Compute MDCM ESP value on grid point
            fvgrid(i) = calc_mdcm_xval(fatom_cxyz, fxgrid(:, i))

        end do

    end subroutine calc_mdcm_grid


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Compute ESP of MDCM model at grid point and with given atom positions
    !
    !///////////////////////////////////////////////////////////////////////////
    real(rp) function calc_mdcm_xval(fatom_cxyz, x)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        real(rp), dimension(:), intent(in)  :: x(3)
        real(rp), dimension(:), intent(in)  :: fatom_cxyz(Nqdim)

        ! Variables
        !-----------

        integer :: i
        real(rp)    :: xval, r

        ! Iterate over MDCM charges
        calc_mdcm_xval = 0._rp
        do i = 1, Nqdim, 4

            ! Compute distance
            r = sqrt(sum((fatom_cxyz(i:i+2) - x)**2))

            ! Check for division by zero
            if(r < eps) r = eps

            ! Add ESP from MDCM charge
            calc_mdcm_xval = calc_mdcm_xval + fatom_cxyz(i + 3)/r

        end do

    end function calc_mdcm_xval


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Compute errors between two value arrays
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine calc_error(Nval, val1, val2, rmse, mae, maxae)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        integer, intent(in) :: Nval
        real(rp), dimension(:), intent(in)  :: val1(Nval), val2(Nval)

        ! Output
        !--------

        real(rp), intent(out)   :: rmse, mae, maxae

        ! Variables
        !-----------

        integer :: i
        real(rp)    :: dval

        ! Iterate over values
        rmse = 0._rp
        mae = 0._rp
        maxae = 0._rp
        do i = 1, Nval

            ! Value deviation
            dval = abs(val1(i) - val2(i))

            ! Add deviation
            mae = mae + dval

            ! Add squared deviation
            rmse = rmse + dval**2

            ! Check for maximum deviation
            if(maxae < dval) maxae = dval

        end do

        ! Compute root of mean deviation
        mae = mae/real(Nval, rp)

        ! Compute root of mean squared value
        rmse = sqrt(rmse/real(Nval, rp))

    end subroutine calc_error


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Compute RMSE between two value arrays
    !
    !///////////////////////////////////////////////////////////////////////////
    real(rp) function calc_rmse(Nval, val1, val2)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        integer, intent(in) :: Nval
        real(rp), dimension(:), intent(in)  :: val1(Nval), val2(Nval)

        ! Variables
        !-----------

        integer :: i

        ! Iterate over values
        calc_rmse = 0._rp
        do i = 1, Nval

            ! Add squared deviation
            calc_rmse = calc_rmse + (val1(i) - val2(i))**2

        end do

        ! Compute root of mean squared value
        calc_rmse = sqrt(calc_rmse/Nval)

    end function calc_rmse


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Compute mean square error between two value arrays
    !
    !///////////////////////////////////////////////////////////////////////////
    real(rp) function calc_mse(Nval, val1, val2)

        implicit none

        integer, parameter :: rp = selected_real_kind(15)   ! python float

        ! Input
        !-------

        integer, intent(in) :: Nval
        real(rp), dimension(:), intent(in)  :: val1(Nval), val2(Nval)

        ! Variables
        !-----------

        integer :: i

        ! Iterate over values
        calc_mse = 0._rp
        do i = 1, Nval

            ! Add squared deviation
            calc_mse = calc_mse + (val1(i) - val2(i))**2

        end do

        ! Compute root of mean squared value
        calc_mse = calc_mse/Nval

    end function calc_mse


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Raise error message
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine raise_error(message)

        implicit none

        character(len=*), intent(in) :: message

        write(*,'(A)') "ERROR: "//message

        call dealloc_all()

        stop

    end subroutine raise_error


    !///////////////////////////////////////////////////////////////////////////
    !
    !   Deallocate stuff
    !
    !///////////////////////////////////////////////////////////////////////////
    subroutine dealloc_all()

        implicit none

        if(allocated(atom_num))     deallocate(atom_num)
        if(allocated(atom_pos))     deallocate(atom_pos)
        if(allocated(atom_lcl))     deallocate(atom_lcl)
        if(allocated(grid_pos))     deallocate(grid_pos)
        if(allocated(grid_val))     deallocate(grid_val)
        if(allocated(grid_aff))     deallocate(grid_aff)
        if(allocated(grid_imp))     deallocate(grid_imp)
        if(allocated(file_val))     deallocate(file_val)
        if(allocated(file_fit))     deallocate(file_fit)
        if(allocated(file_grd))     deallocate(file_grd)
        if(allocated(atom_fit))     deallocate(atom_fit)
        if(allocated(mdcm_cxyz))    deallocate(mdcm_cxyz)
        if(allocated(mdcm_clcl))    deallocate(mdcm_clcl)
        if(allocated(mdcm_poly))    deallocate(mdcm_poly)
        if(allocated(mdcm_afrm))    deallocate(mdcm_afrm)
        if(allocated(mdcm_ftyp))    deallocate(mdcm_ftyp)

    end subroutine dealloc_all

end module dcm_fortran
