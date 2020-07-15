- 创建版本库：

   - 1.选择一个合适的地方，创建一个空目录：
          - 创建一个空目录：mkdir dirname
         - 切换到该目录下：cd dirname 
         - pwd命令用于显示当前目录：pwd 	
         - ls命令用于列出目标目录中所有的子目录和文件：ls 
   - 2.通过git init命令把这个目录变成Git可以管理的仓库：
        - git init

- 添加文件到版本库，一定要放到dirname目录下（子目录也行）：
  	
  	- 1.使用命令git add <filename>，注意，可反复多次使用，添加多个文件；
  	- 2.使用命令git commit -m <message>，完成。

- 查看状态：
  	
  	- 要随时掌握工作区的状态，使用git status命令。
  	- 如果git status告诉你有文件被修改过，用git diff <filename>可以查看修改内容。

- 版本回退：	
  	
  	- HEAD指向的版本就是当前版本，上一个版本就是HEAD^，上上一个版本就是HEAD^^，当然往上100个版本写100个^比较容易数不过来，所以写成HEAD~100。
  	- 因此，Git允许我们在版本的历史之间穿梭，使用命令git reset --hard commit_id。
  	- 穿梭前，用git log可以查看提交历史，以便确定要回退到哪个版本。如果嫌输出信息太多，看的眼花缭乱的，可以试试加上--pretty=oneline参数。
  	- 要重返未来，用git reflog 查看命令历史，以便确定要回到未来的哪个版本。

- 撤销修改：

   - 场景1：当改乱了工作区某个文件的内容，想直接丢弃工作区的修改时，用命令git checkout -- filename。
   - 场景2：当不但乱改了工作区某个文件的内容，还添加到了暂存区时，想丢弃修改，分两步：
     		第一步用命令git reset HEAD <filename>,就回到了场景1，第二步按场景1操作。
   - 场景3：当提交了不合适的修改到版本库时，想要撤销本次提交，用命令git reset --hard HEAD^回退到上一个版本，不过前提是没有推送到远程库。

- 删除文件：

   - 命令rm filename用于在工作区删除一个文件。如果删错了，因为版本库里还有，所以可以很轻松地使用命令git checkout -- filename把误删的文件恢复到最新版本，但是会丢失最近一次提交后修改的内容。
   - 命令git rm filename用于在版本库删除一个文件。

- 添加远程库：

  - 要关联一个远程库，使用命令git remote add origin git@github.com:path/repo-name.git；
    添加后，远程库的名字就是origin。
  - 关联后，使用命令git push -u origin master第一次推送master分支的所有内容到远程库；
  - 此后，每次本地提交后，只要有必要，就可以使用git push origin master推送最新修改；

- 从远程库克隆：

  	- 创建远程库时勾选Initialize this repository with a README，这样GitHub会自动为我们创建一个README.md文件。
  	- 要克隆一个仓库，首先必须知道仓库的地址，然后使用git clone命令克隆。
  	- Git支持多种协议，包括https，但是通过ssh支持的原生git协议速度最快。

- 创建与合并分支：

  	- 查看分支：git branch，这个命令会列出所有分支，当前分支前面会标一个*号。
  	- 创建分支:git branch <name>
  	- 切换分支：git checkout <name>
  	- 创建+切换分支：git checkout -b <name>
  	- 合并某分支到当前分支：git merge <name>
  	- 删除分支:git branch -d <name>

- 解决冲突：

  	- 当Git无法自动合并分支时，就必须首先解决冲突。解决冲突后，再提交，合并完成。
  	- 解决冲突就是把Git合并失败的文件手动编辑为我们希望的内容，再提交。
  	- 用git log --graph命令可以看到分支合并图。

- 分支管理策略：

   - 在实际开发中，我们应该按照几个基本原则进行分支管理：
     - 首先，master分支应该是非常稳定的，也就是仅用来发布新版本，平时不能在上面干活；
     - 那在哪干活呢？干活都在dev分支上，也就是说，dev分支是不稳定的，到某个时候，比如1.0版本发布时，再把dev分支合并到master上，在master分支发布1.0版本；
     - 你和你的小伙伴每个人都在dev分支上干活，每个人都有自己的分支，时不时地往dev分支上合并就可以了。
     - 合并分支时，加上--no--ff参数就可以用普通模式合并，合并后的历史有分支，能看出来曾经做过合并，而fast forward合并就看不出来曾经做过合并。

- BUG分支：

  	- 修复bug时，我们会通过创建新的分支进行修复，然后合并，最后删除；
  	- 当手头工作没有完成时，先把工作现场git stash（储藏）一下，然后去修复bug，修复后，再（恢复工作现场），回到工作现场。
   - 恢复工作现场的方法：
     	- 1.用git stash apply恢复，但是恢复后，stash内容并不删除，需要用git stash drop来删除。
      - 2.用git stash pop恢复，恢复的同时把stash内容也删了。
      - 用命令git stash list 命令查看储藏起来的工作现场。

- Feature分支：

  	- 开发一个新feature，最好新建一个分支；
  	- 如果要丢弃一个没有被合并过的分支，可以通过git branch -D <name>强行删除。

- 多人协作：

   - 查看远程库信息，使用git remote -v;
   - 本地新建的分支如果不推送到远程，对其他人就是不可见的；
   - 推送分支（把该分支上的所有本地提交推送到远程库）：
      - 并不是一定要把本地分支往远程推送，哪些需要，哪些不需要呢？
      - master分支是主分支，因此要时刻与远程同步；
      - dev分支是开发分支，团队所有成员都需要在上面工作，所以也需要与远程同步；
      - bug分支只用于在本地修复bug，就没必要推到远程了；
      - feature分支是否推到远程，取决于你是否和你的小伙伴合作在上面开发。
   - 抓取分支：	
     	- 从本地托推送分支，使用git push origin branch-name，如果推送失败，先用git pull抓取远程的新提交；
     	- 在本地创建和远程分支对应的分支，使用git checkout -b branch-name origin/branch-name，本地和远程分支的名称最好一致；
     	- 建立本地分支和远程分支的关联，使用git branch --set-upstream branch-name origin/branch-name；
     	- 从远程抓取分支，使用git pull，如果有冲突，要先处理冲突。
   - 因此，多人协作的工作模式通常是这样：
     	- 1.首先，可以试图git push origin <branch-name>推送自己的修改；
     	- 2.如果推送失败，则因为远程分支比你的本地更新，需要先用git pull试图合并；
     	- 3.如果合并有冲突，则可以解决冲突，并在本地提交；
     	- 4.没有冲突或者解决掉冲突后，再用git push origin <branch-name>推送就能成功！
     	- 如果git pull提示no tracking information,则说明本地dev分支和远程origin/dev分支的链接关系没有创建，用命令git branch --set-upstream-to <branch-name> origin/<branch-name>。

- Rebase:

  	- rebase操作可以把本地未push的分支提交历史整理成直线；
  	- rebase的目的是使得我们在查看历史提交的变化时更容易，因为分叉的提交需要三方对比。

- 创建标签：

  	- 命令git tag <tagname>用于新建一个标签，默认为HEAD,也可以指定一个commit id；
   - 命令git tag -a <tagname> -m "blablabla..."可以指定标签信息；
   - 命令git tag可以查看所有标签；
   - 命令git show <tagname>查看标签信息，看到说明文字。

- 操作标签：

  	- 命令git push origin <tagname>可以推送一个本地标签；
  	- 命令git push origin --tags可以推送全部未推送过的本地标签；
  	- 命令git tag -d <tagname>可以删除一个本地标签；
   - 如果标签已经推送到远程，先从本地删除，再从远程删除，用命令git push origin :refs/tags/<tagname>可以删除一个远程标签。

- 使用GitHub:

    - 在GitHub上，可以任意Fork开源仓库；
    - 自己拥有Fork后的仓库的读写权限；
    - 可以推送pull request给官方仓库来贡献代码。

- 自定义Git:

  	- 让Git显示颜色，会让命令输出看起来更醒目：git config --global color.ui true，这样，Git会适当显示不同的颜色。

- 忽略特殊文件：

  	- 忽略某些文件时，需要编写.gitignore；
  	- .gitignore文件本身要放到版本库里，并且可以对.gitignore做版本管理；
  	- 如果发现.gitignore写得有问题，需要找出来到底哪个规则写错了，可以用git check-ignore命令检查。
  	- 有些时候，想添加一个文件到Git，但发现添加不了，原因是这个文件被.gitignore忽略了，可以用-f强制添加到Git。

- 配置别名：

  	- git config --global alias.st status 以后st就代表status。
  	- 配置Git的时候，加上--global是针对当前用户起作用的，如果不加，那只针对当前的仓库起作用。
  	- 每个仓库的配置文件都放在.git/config文件中，别名就在[alias]后面，要删除别名，直接把对应的行删掉即可。
  	- 而当前用户的Git配置文件放在用户主目录下的一个隐藏文件.gitconfig中，配置别名也可以直接修改这个文件，如果改错了，可以删掉文件重新通过命令配置。

- ```
  git add .
  git commit -m "first commit"
  git push origin master	
  ```

- 第一次推送到远程仓库时，要先git pull然后git push -f origin master，提示输入用户名和密码

  ```
  git init
  git rm -r --cached .
  git config core.autocrlf false
  git add .
  git commit -m "first commit"	
  git remote add origin https://github.com/Crystalgirl211/NumericalAnalysis.git	
  git push -u origin master
  ```

  

