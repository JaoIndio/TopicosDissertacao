let g:newtrw_banner = 0 
let g:newtr_alto    = 1 
let g:newtr_preview = 0
let g:netrw_altv    = 1
let g:newtrw_liststyle = 3  
"tree view

set number
syntax on
set tabstop=2
set shiftwidth=2
set expandtab
set softtabstop=2
set smarttab
set smartindent
set autoindent
 
set ai

colorscheme darkblue
 
au BufNewFile,BufRead *.puml set filetype=plantuml
colorscheme darkblue

"syn match VariableName "\<[a-zA-Z_]\w*\>" 
"syn match FunctionName "\<[a-zA-Z_]\w*\>("

"hi FunctionName ctermfg=red
"hi VariableName ctermfg=blue


set hlsearch

set laststatus=2
set statusline+=%f
set statusline+=\ %{&number?'':printf('%2d,',line('.'))} " Line numbera
set statusline+=%-2v " Virtual column numbera

hi statusline ctermfg=9 ctermbg=232

set tags=$HOME/esp/tags,$HOME/.arduino15/packages/esp32/hardware/tags,./tags

"" $HOME/curso/AufTek/ArcoGen/PVAG_Modbus/tags
"" $HOME/curso/AufTek/uTower/projetos/FreeRTOS_C4D_Ver15.0/AufTek_dev/Wifi_reconect/tags,./tags,
"" $HOME/.arduino15/packages/esp32/hardware/tags,
"" $HOME/curso/AufTek/uTower/projetos/FreeRTOS_C4D_Ver15.0/tags,
"" $HOME/curso/AufTek/freeRTOS_curso/tags,/tags,
"" $HOME/curso/AufTek/uTower/projetos/tags,
"" $HOME/curso/AufTek/uTower_Firmware/tags,tags,
"" $HOME/tags 

"" Search down into subfolders
"" Provides tab-completion for all files--related tasks
set path+=**

"" Display all mathcing files when we tab complete
set wildmenu

map <C-\> :tab split<CR>:exec("tag ".expand("<cword>"))<CR>
map <A-]> :vsp <CR>:exec("tag ".expand("<cword>"))<CR>

hi! Normal guibg=NONE ctermbg=NONE



set makeprg=idf.py\ -p\ /dev/ttyUSB0\ flash\ monitor\
  
"hi CursorLine ctermbg=196 
"
augroup changeCursor
"" Chama a função ao entrar em uma janela
  autocmd!
  autocmd WinEnter * set cursorline
  autocmd WinLeave * set nocursorline

augroup END

function! GitBranch()
  let branchname = substitute(system('git rev-parse --abbrev-ref HEAD'), '\n', '', '')
  return strlen(branchname) ? ' (' . branchname . ')' : ''
endfunction


let &statusline .=' %{GitBranch()}'










":tabeedit <path/filename> : abre nova tab
":tabn : vai para 'next' tab
":tabp : vai para 'previous' tab
":tabn NUM : vai para tab NUM
" gT go to left tab
" gt go to right tab
" 1gt, 2gt, 3gt go first, second, third... tab
" 
"search: *
