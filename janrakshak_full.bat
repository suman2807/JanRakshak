@echo off
setlocal enabledelayedexpansion

for %%m in (09 10 11 12) do (
    for /L %%d in (1,1,31) do (
        REM skip days after 15 Dec
        if not %%m==12 (
            REM main logic for months other than December
            set processDay=true
        ) else (
            if %%d LEQ 15 (
                set processDay=true
            ) else (
                set processDay=false
            )
        )
        if /i !processDay!==true (
            REM activity chance
            set /a chance=!random! %% 10
            if !chance! GEQ 4 (
                REM 1–5 commits
                set /a commits=!random! %% 5 + 1
                for /L %%c in (1,1,!commits!) do (
                    set day=%%d
                    if !day! LSS 10 set day=0!day!
                    set /a hour=!random! %% 12 + 9
                    set /a min=!random! %% 60
                    echo %%m-%%d-%%c >> file.txt
                    git add .
                    set DATE=2024-%%m-!day! !hour!:!min!:00
                    set GIT_AUTHOR_DATE=!DATE!
                    set GIT_COMMITTER_DATE=!DATE!
                    REM September
                    if %%m==09 (
                        set /a r=!random! %% 4
                        if !r! EQU 0 git commit -m "initialized project"
                        if !r! EQU 1 git commit -m "added basic UI"
                        if !r! EQU 2 git commit -m "setup backend"
                        if !r! EQU 3 git commit -m "updated README"
                    )
                    REM October
                    if %%m==10 (
                        set /a r=!random! %% 4
                        if !r! EQU 0 git commit -m "implemented authentication"
                        if !r! EQU 1 git commit -m "added alert feature"
                        if !r! EQU 2 git commit -m "connected APIs"
                        if !r! EQU 3 git commit -m "fixed login bugs"
                    )
                    REM November
                    if %%m==11 (
                        set /a r=!random! %% 4
                        if !r! EQU 0 git commit -m "added notifications"
                        if !r! EQU 1 git commit -m "integrated maps"
                        if !r! EQU 2 git commit -m "optimized performance"
                        if !r! EQU 3 git commit -m "refactored code"
                    )
                    REM December
                    if %%m==12 (
                        set /a r=!random! %% 4
                        if !r! EQU 0 git commit -m "final improvements"
                        if !r! EQU 1 git commit -m "fixed critical bugs"
                        if !r! EQU 2 git commit -m "improved UI"
                        if !r! EQU 3 git commit -m "updated documentation"
                    )
                )
            )
        )
    )
)