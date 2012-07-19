# To help simplify including Clique in external projects
include @CMAKE_INSTALL_PREFIX@/conf/cliqvariables

PSP_COMPILE_FLAGS = ${CLIQ_COMPILE_FLAGS}
PSP_LINK_FLAGS = ${CLIQ_LINK_FLAGS}

PSP_LIBS = -lpsp ${CLIQ_LIBS}
